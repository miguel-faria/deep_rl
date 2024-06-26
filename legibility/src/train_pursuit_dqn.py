#! /usr/bin/env python

import os
import sys
import argparse
import numpy as np
import flax.linen as nn
import math
import random
import jax
import optax
import time
import json

from dl_algos.dqn import EPS_TYPE
from dl_algos.multi_model_madqn import MultiAgentDQN
from dl_envs.pursuit.pursuit_env import PursuitEnv, Action
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from termcolor import colored
from typing import List, Callable
from flax.training.train_state import TrainState
from datetime import datetime

RNG_SEED = 13042023
TEST_RNG_SEED = 12072023
ACTION_DIM = 5


class PreyAgent(object):
	
	_decision_mode: int
	_decision_func: Callable
	
	def __init__(self, decision_mode: int, decision_func: Callable = None):
		self._decision_mode = decision_mode
		self._decision_func = decision_func
	
	def act(self, obs) -> int:
		if self._decision_mode == 0:
			return np.random.choice(ACTION_DIM)
		else:
			return self._decision_func(obs)


class StayPrey(PreyAgent):
	
	def act(self, obs) -> int:
		return Action.STAY


class OccasionalMovePrey(PreyAgent):
	
	_move_prob: float
	_rng_gen: np.random.Generator
	
	def __init__(self, decision_mode: int, decision_func: Callable = None, move_prob: float = 0.1, rng_seed: int = 123456789):
		super().__init__(decision_mode, decision_func)
		self._move_prob = move_prob
		self._rng_gen = np.random.default_rng(rng_seed)
	
	def act(self, obs) -> int:
		if self._rng_gen.random() > self._move_prob:
			return np.random.choice(ACTION_DIM - 1)
		else:
			return Action.STAY


def get_history_entry(obs: np.ndarray, actions: List[int], hunter_ids: List[str]) -> List:
	entry = []
	for hunter in hunter_ids:
		a_idx = hunter_ids.index(hunter)
		state_str = ' '.join([str(x) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def train_dtde_pursuit_model(agents_ids: List[str], pursuit_env: PursuitEnv, pursuit_models: MultiAgentDQN, prey_models: List[Callable], num_iterations: int,
							 max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float, eps_type: str,
							 rng_seed: int, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 10, target_freq: int = 1000,
							 summary_freq: int = 1000) -> List:
	def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, step: int, max_steps: int):
		
		if update_type == 1:
			return max(((final_eps - init_eps) / (max_steps * decay_rate)) * step + init_eps, end_eps)
		elif update_type == 2:
			return max(decay_rate ** step * init_eps, end_eps)
		elif update_type == 3:
			return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)
		elif update_type == 4:
			return max((decay_rate * math.sqrt(step)) * init_eps, end_eps)
		else:
			print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
			return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)
	
	# recorded_obs = set()
	history = []
	random.seed(rng_seed)
	np.random.seed(rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	key = jax.random.PRNGKey(rng_seed)
	key, q_key = jax.random.split(key, 2)
	
	obs, _, _, _ = pursuit_env.reset()
	for hunter in pursuit_models.agent_ids:
		pursuiter_dqn = pursuit_models.agent_dqns[hunter]
		hunter_idx = pursuit_models.agent_ids.index(hunter)
		if pursuiter_dqn.online_state is None:
			pursuiter_dqn.online_state = TrainState.create(
				apply_fn=pursuiter_dqn.q_network.apply,
				params=pursuiter_dqn.q_network.init(q_key, obs[hunter_idx]),
				tx=optax.adam(learning_rate=optim_learn_rate),
			)
		if pursuiter_dqn.target_params is None:
			pursuiter_dqn.target_params = pursuiter_dqn.q_network.init(q_key, obs[hunter_idx])
	
		pursuiter_dqn.q_network.apply = jax.jit(pursuiter_dqn.q_network.apply)
		pursuiter_dqn.target_params = optax.incremental_update(pursuiter_dqn.online_state.params, pursuiter_dqn.target_params, 1.0)
	
	start_time = time.time()
	epoch = 0
	n_agents = len(agents_ids)
	n_hunters = len(pursuit_models.agent_ids)
	episode_rewards = [0] * n_hunters
	episode_start = epoch
	
	for it in range(num_iterations):
		print("Iteration %d out of %d" % (it + 1, num_iterations))
		done = False
		episode_history = []
		while not done:
			print("Epoch %d" % (epoch + 1))
			
			# interact with environment
			if eps_type == 'linear':
				eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			else:
				eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			if rng_gen.random() < eps:
				actions = []
				prey_idx = 0
				for a_id in agents_ids:
					if a_id.find('prey') != -1:
						actions += [prey_models[prey_idx](obs[0])]
						prey_idx += 1
					else:
						actions += [rng_gen.choice(ACTION_DIM)]
			else:
				actions = []
				prey_idx = 0
				for a_id in agents_ids:
					if a_id.find('prey') != -1:
						actions += [prey_models[prey_idx](obs[0])]
						prey_idx += 1
					else:
						agent_dqn = pursuit_models.agent_dqns[a_id]
						hunter_idx = pursuit_models.agent_ids.index(a_id)
						q_values = agent_dqn.q_network.apply(agent_dqn.online_state.params, obs[hunter_idx])
						action = q_values.argmax(axis=-1)
						actions += [int(jax.device_get(action))]
						if pursuit_models.agent_dqns[a_id].use_tracker and epoch % summary_freq == 0:
							pursuit_models.agent_dqns[a_id].summary_writer.add_scalar("charts/episodic_q_vals", float(q_values[int(action)]), epoch)
			next_obs, rewards, finished, infos = pursuit_env.step(actions)
			episode_history += [get_history_entry(obs, actions, pursuit_models.agent_ids)]
			
			# store new samples and update episode rewards
			for hunter in pursuit_models.agent_ids:
				hunter_idx = pursuit_models.agent_ids.index(hunter)
				a_idx = agents_ids.index(hunter)
				pursuit_models.agent_dqns[hunter].replay_buffer.add(obs[hunter_idx], next_obs[hunter_idx], np.array(actions[a_idx]),	# store examples
																	rewards[a_idx], finished, infos)
				episode_rewards[hunter_idx] += rewards[a_idx]
				if pursuit_models.agent_dqns[hunter].use_tracker:
					if finished:
						print(hunter, rewards[a_idx], episode_rewards[hunter_idx])
					pursuit_models.agent_dqns[hunter].summary_writer.add_scalar("charts/reward", rewards[a_idx], epoch)
					pursuit_models.agent_dqns[hunter].summary_writer.add_text("logs/observation", str(obs[hunter_idx]), epoch)
					pursuit_models.agent_dqns[hunter].summary_writer.add_text("logs/action", str(actions[hunter_idx]), epoch)
					pursuit_models.agent_dqns[hunter].summary_writer.add_text("logs/next_observation", str(next_obs[hunter_idx]), epoch)
			
			obs = next_obs
			
			# update Q-network and target network
			if epoch > warmup:
				if epoch % train_freq == 0:
					pursuit_models.update_models(batch_size, epoch, start_time, summary_freq)
				
				if epoch % target_freq == 0:
					for hunter in pursuit_models.agent_ids:
						pursuit_models.agent_dqns[hunter].target_params = optax.incremental_update(pursuit_models.agent_dqns[hunter].online_state.params,
																								   pursuit_models.agent_dqns[hunter].target_params, tau)
				
			epoch += 1
			sys.stdout.flush()
			if finished:
				for hunter in pursuit_models.agent_ids:
					hunter_idx = pursuit_models.agent_ids.index(hunter)
					if pursuit_models.agent_dqns[hunter].use_tracker:
						pursuit_models.agent_dqns[hunter].summary_writer.add_scalar("charts/episodic_return", episode_rewards[hunter_idx], it)
						pursuit_models.agent_dqns[hunter].summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it)
						pursuit_models.agent_dqns[hunter].summary_writer.add_scalar("charts/epsilon", eps, epoch)
				pursuit_env.reset_init_pos()
				obs, _, _, _ = pursuit_env.reset()
				episode_rewards = [0] * n_hunters
				episode_start = epoch
				history += [episode_history]
				done = True
	
	return history


def main():
	parser = argparse.ArgumentParser(description='Train DQN for Pursuit environment')
	
	# Multi-agent DQN params
	parser.add_argument('--nlayers', dest='n_layers', type=int, required=True, help='Number of layers for the neural net in the DQN')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	
	# Train parameters
	parser.add_argument('--cycles', dest='n_cycles', type=int, required=True,
						help='Number of training cycles, each cycle spawns the field with a different configuration of hunters and preys.')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.95, help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
						help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	
	# Environment parameters
	parser.add_argument('--hunters', dest='hunters', type=str, nargs='+', required=True, help='IDs of hunters in the environment')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--preys', dest='preys', type=str, nargs='+', required=True, help='IDs of preys in the environment')
	parser.add_argument('--n-catch', dest='n_catch', type=int, required=True, help='Number of hunters that have to surround the prey to catch it')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	
	args = parser.parse_args()
	n_layers = args.n_layers
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_vdn = args.use_vdn
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	n_cycles = args.n_cycles
	n_iterations = args.n_iterations
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	learn_rate = args.learn_rate
	target_learn_rate = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	hunters = args.hunters
	field_lengths = args.field_lengths
	preys = args.preys
	n_catch = args.n_catch
	max_steps = args.max_steps
	
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	if not use_gpu:
		jax.config.update('jax_platform_name', 'cpu')
	
	# print(gamma, initial_eps, final_eps, eps_decay, eps_type, warmup, learn_rate, target_learn_rate)
	field_dims = len(field_lengths)
	if 2 >= field_dims > 0:
		if field_dims == 1:
			field_size = (field_lengths[0], field_lengths[0])
			sight = field_lengths[0]
		else:
			field_size = (field_lengths[0], field_lengths[1])
			sight = max(field_lengths[0], field_lengths[1])
	else:
		print('[ARGS ERROR] Field size must either be composed of only 1 or 2 arguments; %d were given. Exiting program' % field_dims)
		return
	
	n_hunters = len(hunters)
	n_preys = len(preys)
	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = ('train_pursuit_dqn_%dx%d-field_%d-hunters_%d-catch' % (field_size[0], field_size[1], n_hunters, n_catch)) + '_' + now.strftime("%Y%m%d-%H%M%S")
	model_path = (models_dir / 'pursuit_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-hunters' % n_hunters)) / now.strftime("%Y%m%d-%H%M%S")
	
	sys.stdout = open(log_dir / (log_filename + '_log.txt'), 'a')
	sys.stderr = open(log_dir / (log_filename + '_err.txt'), 'w')
	
	print('##########################')
	print('Starting Pursuit DQN Train')
	print('##########################')
	print('Environment setup')
	env = PursuitEnv(hunters, preys, field_size, n_hunters, n_catch, max_steps)
	env.seed(RNG_SEED)
	rng_gen = np.random.default_rng(RNG_SEED)
	
	print('Setup multi-agent DQN')
	obs_dims = [field_size[0], field_size[1], 2, 2, n_hunters + 1] * (n_hunters + n_preys)
	agents_dqns = MultiAgentDQN(n_hunters, hunters, len(Action), n_layers, nn.relu, layer_sizes, buffer_size, gamma, MultiDiscrete(obs_dims),
								use_gpu, dueling_dqn, use_ddqn, False, use_tensorboard, tensorboard_details)

	print('Starting train')
	sys.stdout.flush()
	# prey = PreyAgent(0)
	prey = StayPrey(0)
	history = train_dtde_pursuit_model(hunters + preys, env, agents_dqns, [prey.act], n_iterations, n_iterations * max_steps, batch_size, learn_rate,
									   target_learn_rate, initial_eps, final_eps, eps_type, RNG_SEED, eps_decay, warmup, train_freq, target_freq,
									   tensorboard_freq)

	Path.mkdir(model_path, parents=True, exist_ok=True)
	agents_dqns.save_models(('%d-catch' % n_catch), model_path)
	obs_path = model_path / ('%d-catch.json' % n_catch)
	with open(obs_path, "w") as of:
		of.write(json.dumps(history))
	sys.stdout.flush()
	
	print('Testing trained model')
	env.seed(TEST_RNG_SEED)
	rng_gen = np.random.default_rng(TEST_RNG_SEED)
	np.random.seed(TEST_RNG_SEED)
	init_pos_hunter = {}
	for hunter in hunters:
		hunter_idx = hunters.index(hunter)
		init_pos_hunter[hunter] = (hunter_idx // n_hunters, hunter_idx % n_hunters)
	init_pos_prey = {}
	for prey in preys:
		prey_idx = preys.index(prey)
		init_pos_prey[prey] = (max(field_size[0] - (prey_idx // n_preys) - 1, 0), max(field_size[1] - (prey_idx % n_preys) - 1, 0))
	env.spawn_hunters(init_pos_hunter)
	env.spawn_preys(init_pos_prey)
	obs, _, _, _ = env.reset()
	epoch = 0
	history = []
	game_over = False
	print(env.field)
	while not game_over:
		
		actions = []
		for a_id in agents_dqns.agent_ids:
			agent_dqn = agents_dqns.agent_dqns[a_id]
			a_idx = agents_dqns.agent_ids.index(a_id)
			q_values = agent_dqn.q_network.apply(agent_dqn.online_state.params, obs[a_idx])
			action = q_values.argmax(axis=-1)
			action = jax.device_get(action)
			actions += [action]
		actions = np.array(actions)
		print(' '.join([str(Action(action)) for action in actions]))
		next_obs, rewards, finished, infos = env.step(actions)
		history += [get_history_entry(obs, actions, hunters)]
		obs = next_obs
		print(env.field)
		
		if finished or epoch >= max_steps:
			game_over = True
		
		sys.stdout.flush()
		epoch += 1
	
	print('Epochs needed to finish: %d' % epoch)
	print('Test history:')
	print(history)


if __name__ == '__main__':
	main()
