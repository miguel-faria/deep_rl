#! /usr/bin/env python

import gymnasium
import jax
import numpy as np
import flax.linen as nn
import jax.numpy as jnp
import wandb

from algos.ddpg import DDPG
from utilities.buffers import ReplayBuffer
from datetime import datetime
from pathlib import Path


N_ITERATIONS = 5000
LEARN_RATE = 1e-4


def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, step: int, max_steps: int):
	if update_type == 1:
		return max(((end_eps - init_eps) / max_steps) * step / decay_rate + init_eps, end_eps)
	elif update_type == 2:
		return max(decay_rate ** step * init_eps, end_eps)
	elif update_type == 3:
		return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)
	else:
		return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)


def train_ddpg(wandb_run: wandb.run, env_id: str):
	
	env = gymnasium.make(env_id) #, render_mode='human')
	action_bounds = (env.action_space.low[0], env.action_space.high[0])
	action_scale = jnp.array((action_bounds[1] - action_bounds[0]) / 2.0)
	action_bias = jnp.array((action_bounds[1] + action_bounds[0]) / 2.0)
	use_gpu = True
	rng_seed = 1245
	buffer_size = 100000
	ddpg = DDPG(action_dim=len(env.action_space.shape), num_layers=2, act_function=nn.relu, layer_sizes=[256, 256], action_bias=action_bias,
				action_scale=action_scale, use_wandb=True, wandb_writer=wandb_run, action_bounds=action_bounds)
	buffer = ReplayBuffer(buffer_size, env.observation_space, env.action_space, "cuda" if use_gpu else "cpu", handle_timeout_termination=False, rng_seed=rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	obs, *_ = env.reset(seed=rng_seed)
	sample_actions = env.action_space.sample()
	ddpg.init_networks(rng_seed, obs, sample_actions, actor_lr=LEARN_RATE, critic_lr=LEARN_RATE)
	init_eps = 1.0
	final_eps = 0.05
	warmup = 2000
	batch_size = 500
	gamma = 0.99
	tau = 0.1
	decay_rate = 0.6
	train_freq = 1
	target_freq = 2
	epoch = 0
	episode_lens = []
	
	for it in range(N_ITERATIONS):
		
		done = False
		episode_reward = 0
		avg_critic_loss = 0
		avg_actor_loss = 0
		episode_len = 0
		episod_start = epoch
		eps = eps_update(1, init_eps, final_eps, decay_rate, it, max_steps=N_ITERATIONS)
		while not done:
		
			if epoch <= warmup:
				action = np.clip(env.action_space.sample(), action_bounds[0], action_bounds[1])
			else:
				action = jax.device_get(ddpg.actor_network.apply(ddpg.actor_online_state.params, obs)[0])
				action = [np.clip(action + rng_gen.normal(0, ddpg.actor_network.action_scale * eps), action_bounds[0], action_bounds[1])]

			action = np.array(action)
			next_obs, reward, finished, timeout, info = env.step(action)
			# reward = -1 if (reward < 0) else reward

			buffer.add(obs, next_obs, action, reward, finished, info)
			episode_reward += reward
			obs = next_obs
			epoch = epoch + 1

			if epoch >= warmup:
				if it % train_freq == 0:
					critic_loss, agent_loss = jax.device_get(ddpg.update_models(buffer, batch_size, gamma, None))
					avg_critic_loss += critic_loss
					avg_actor_loss += agent_loss

				if it % target_freq == 0:
					ddpg.update_targets(tau)
					
			if finished or timeout:
				done = True
				episode_len = epoch - episod_start
				episode_lens.append(episode_len)
				ddpg.wandb_writer.log({
					"charts/performance/episode_return": episode_reward,
					"charts/performance/mean_episode_return": episode_reward / episode_len,
					"charts/performance/episode_length": episode_len,
					"charts/performance/avg_episode_length": np.mean(episode_lens),
					"charts/control/iteration": it,
					"charts/control/epsilon": eps},
					step=it)
				obs, *_ = env.reset()

		ddpg.wandb_writer.log({
				"loss/average_critic_loss":               avg_critic_loss / episode_len,
				"loss/average_actor_loss":                avg_actor_loss / episode_len},
				step=it)


if __name__ == '__main__':
	try:
		now = datetime.now()
		env_name = 'MountainCarContinuous-v0'
		# env_name = 'Pendulum-v1'
		run = wandb.init(project='ddpg-trials', entity='miguel-faria',
				   config={
					   "env_name": env_name,
					   "num_iterations": N_ITERATIONS,
					   "actor_lr": LEARN_RATE,
					   "critic_lr": LEARN_RATE,
				   },
				   name=('ddpg_test' + now.strftime("%Y%m%d-%H%M%S")))
		train_ddpg(run, env_name)
		wandb.finish()
	
	except KeyboardInterrupt:
		wandb.finish()
