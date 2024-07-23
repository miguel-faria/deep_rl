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


N_ITERATIONS = 200
LEARN_RATE = 3e-4


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
	
	# env = gymnasium.make('MountainCarContinuous-v0', render_mode='human')
	env = gymnasium.make(env_id, g=9.81, render_mode='human')
	action_scale = jnp.array((env.action_space.high - env.action_space.low) / 2.0)
	action_bias = jnp.array((env.action_space.high + env.action_space.low) / 2.0)
	use_gpu = True
	rng_seed = 1245
	buffer_size = 3000
	ddpg = DDPG(action_dim=len(env.action_space.shape), num_layers=2, act_function=nn.relu, layer_sizes=[128, 128], action_bias=action_bias,
				action_scale=action_scale, use_wandb=True, wandb_writer=wandb_run)
	buffer = ReplayBuffer(buffer_size, env.observation_space, env.action_space, "cuda" if use_gpu else "cpu", handle_timeout_termination=False, rng_seed=rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	obs, *_ = env.reset()
	sample_actions = env.action_space.sample()
	ddpg.init_networks(rng_seed, obs, sample_actions, actor_lr=LEARN_RATE, critic_lr=LEARN_RATE)
	init_eps = 1.0
	final_eps = 0.05
	warmup = 400
	batch_size = 32
	gamma = 0.99
	tau = 0.1
	decay_rate = 0.2
	train_freq = 1
	target_freq = 10
	epoch = 0
	episode_lens = []
	
	for it in range(N_ITERATIONS):
		
		done = False
		episode_reward = 0
		avg_critic_loss = 0
		avg_actor_loss = 0
		episod_start = epoch
		eps = eps_update(3, init_eps, final_eps, decay_rate, it, max_steps=N_ITERATIONS)
		while not done:
		
			if rng_gen.random() < eps:
				action = env.action_space.sample()
			else:
				action = [jax.device_get(ddpg.actor_network.apply(ddpg.actor_online_state.params, obs)[0])]
			
			action = np.array(action)
			next_obs, reward, finished, timeout, info = env.step(action)
			
			buffer.add(obs, next_obs, action, reward, finished, info)
			episode_reward += reward
			obs = next_obs
			
			if epoch >= warmup:
				if epoch % train_freq == 0:
					critic_loss, agent_loss = jax.device_get(ddpg.update_models(buffer, batch_size, gamma, None))
					avg_critic_loss += critic_loss
					avg_actor_loss += agent_loss
				
				if epoch % target_freq == 0:
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
					"charts/control/epsilon": eps,
					"loss/average_critic_loss": avg_critic_loss / episode_len,
					"loss/average_actor_loss": avg_actor_loss / episode_len},
					step=it)
				obs, *_ = env.reset()
				
			epoch = epoch + 1
			

if __name__ == '__main__':
	try:
		now = datetime.now()
		# env_name = 'MountainCarContinuous-v0'
		env_name = 'Pendulum-v1'
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
