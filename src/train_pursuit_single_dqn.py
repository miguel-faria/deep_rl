#! /usr/bin/env python

import os
import sys
import argparse
import numpy as np
import flax.linen as nn
import gymnasium
import jax
import json
import math
import time
import logging

from dl_algos.single_model_madqn import SingleModelMADQN
from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_envs.pursuit.pursuit_env import PursuitEnv
from dl_envs.pursuit.agents.agent import Agent, AgentType
from dl_envs.pursuit.agents.target_agent import TargetAgent
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from itertools import permutations
from typing import List, Tuple
from datetime import datetime


RNG_SEED = 6102023
TEST_RNG_SEED = 4072023
N_TESTS = 100


def eps_cycle_schedule(schedule_type: str, cycle_nr: int, max_cycles: int, init_eps: float, final_eps: float, decay_rate: float) -> float:
	if schedule_type == 'log':
		return max(init_eps * (1 / (1 + decay_rate * cycle_nr)), final_eps)
	elif schedule_type == 'exp':
		return max(init_eps - decay_rate ** ((max_cycles - 1) / cycle_nr), final_eps)
	else:
		return max(init_eps * (decay_rate ** cycle_nr), final_eps)


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def get_target_seqs(targets: List[str]) -> List[Tuple[str]]:
	
	if len(targets) > 0:
		return list(permutations(targets, len(targets)))
	else:
		return None


def train_pursuit_dqn(dqn_model: SingleModelMADQN, env: PursuitEnv, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float,
					  tau: float, initial_eps: float, final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99,
					  warmup: int = 0, train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1,
					  use_render: bool = False, cycle: int = 0, greedy_action: bool = True) -> List:
	rng_gen = np.random.default_rng(rng_seed)
	
	history = []
	# Setup DQNs for training
	obs, *_ = env.reset()
	dqn_model.agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
	
	for hunter_id in env.hunter_ids:
		if isinstance(env.agents[hunter_id], TargetAgent):
			env.set_target(hunter_id, env.prey_ids[rng_gen.integers(0, env.n_preys)])
	
	start_time = time.time()
	epoch = 0
	sys.stdout.flush()
	start_record_it = cycle * num_iterations
	start_record_epoch = cycle * max_timesteps
	model_history = []
	
	for it in range(num_iterations):
		if use_render:
			env.render()
		done = False
		episode_rewards = 0
		episode_start = epoch
		episode_history = []
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		preys_alive = env.n_preys
		while not done:
			
			# interact with environment
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			if rng_gen.random() < eps:
				actions = np.array(env.action_space.sample())
			else:
				actions = []
				for a_idx in range(env.n_hunters):
					if dqn_model.agent_dqn.cnn_layer:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
					else:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, obs[a_idx])
					if greedy_action:
						action = q_values.argmax(axis=-1)
					else:
						pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
						pol = pol / pol.sum()
						action = rng_gen.choice(range(env.action_space[0].n), p=pol)
					action = jax.device_get(action)
					if dqn_model.agent_dqn.use_summary and epoch % tensorboard_frequency == 0:
						dqn_model.agent_dqn.summary_writer.add_scalar("charts/episodic_q_vals", float(q_values[int(action)]), epoch + start_record_epoch)
					actions += [action]
				for prey_id in env.prey_ids:
					actions += [env.agents[prey_id].act(env)]
				actions = np.array(actions)
			episode_history += [dqn_model.get_history_entry(obs, actions)]
			next_obs, rewards, terminated, timeout, infos = env.step(actions)
			for a_idx in range(env.n_hunters):
				rewards[a_idx] = env.agents[env.hunter_ids[a_idx]].get_reward(rewards[a_idx], env=env)
			if use_render:
				env.render()
			
			if len(rewards) == 1:
				rewards = np.array([rewards] * dqn_model.num_agents)
			
			if 0 < env.n_preys < preys_alive:
				for hunter_id in env.hunter_ids:
					if isinstance(env.agents[hunter_id], TargetAgent):
						env.set_target(hunter_id, env.prey_ids[rng_gen.integers(0, env.n_preys)])
				preys_alive -= 1
			
			if terminated:
				finished = np.ones(dqn_model.num_agents)
			else:
				finished = np.zeros(dqn_model.num_agents)
			
			# store new samples
			real_next_obs = list(next_obs).copy()
			for a_idx in range(env.n_hunters):
				dqn_model.agent_dqn.replay_buffer.add(obs[a_idx], real_next_obs[a_idx], actions[a_idx], rewards[a_idx], finished[a_idx], [infos])
				episode_rewards += (rewards[a_idx] / dqn_model.num_agents)
			if dqn_model.agent_dqn.use_summary:
				dqn_model.agent_dqn.summary_writer.add_scalar("charts/reward", sum(rewards), epoch + start_record_epoch)
			obs = next_obs
			
			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					dqn_model.update_model(batch_size, epoch, start_time, tensorboard_frequency, logger)
				
				if epoch % target_freq == 0:
					dqn_model.agent_dqn.update_target_model(tau)
			
			epoch += 1
			sys.stdout.flush()
			
			# Check if iteration is over
			if terminated or timeout:
				if dqn_model.write_tensorboard:
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it + start_record_it)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it + start_record_it)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
				env.reset_init_pos()
				obs, *_ = env.reset()
				done = True
				history += [episode_history]
				episode_rewards = 0
				episode_start = epoch
					
	return history


# noinspection DuplicatedCode
def main():
	parser = argparse.ArgumentParser(description='Train DQN for LB Foraging with fixed foods in environment')
	
	# Multi-agent DQN params
	parser.add_argument('--nagents', dest='n_agents', type=int, required=True, help='Number of agents in the environment')
	parser.add_argument('--nlayers', dest='n_layers', type=int, required=True, help='Number of layers for the neural net in the DQN')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	
	# Train parameters
	# parser.add_argument('--cycles', dest='n_cycles', type=int, default=0,
	# 					help='Number of training cycles, each cycle spawns the field with a different food items configurations.')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.5, help='Decay rate for the exploration update')
	parser.add_argument('--cycle-eps-decay', dest='cycle_eps_decay', type=float, required=False, default=0.95, help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
						help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	parser.add_argument('--restart', dest='restart_train', action='store_true',
						help='Flag that signals that train is suppose to restart from a previously saved point.')
	parser.add_argument('--restart-info', dest='restart_info', type=str, nargs='+', required=False, default=None,
						help='List with the info required to recover previously saved model and restart from same point: '
							 '<model_dirname: str> <model_filename: str> <last_cycle: int> Use only in combination with --restart option')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	parser.add_argument('--train-targets', dest='train_targets', type=str, nargs='+', required=False, default=None,
						help='List with the prey ids to train to catch')
	
	# Environment parameters
	parser.add_argument('--hunter-ids', dest='hunter_ids', type=str, nargs='+', required=True, help='List with the hunter ids in the environment')
	parser.add_argument('--prey-ids', dest='prey_ids', type=str, nargs='+', required=True, help='List with the prey ids in the environment')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--hunter-classes', dest='hunter_class', type=int, required=True, help='Class of agent to use for the hunters')
	parser.add_argument('--n-hunters-catch', dest='require_catch', type=int, required=True, help='Minimum number of hunters required to catch a prey')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag that signals the use of the field render while training')
	
	args = parser.parse_args()
	# DQN args
	n_agents = args.n_agents
	n_layers = args.n_layers
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_cnn = args.use_cnn
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	
	# Train args
	n_iterations = args.n_iterations
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	learn_rate = args.learn_rate
	target_update_rate = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	debug = args.debug
	cycle_decay = args.cycle_eps_decay
	
	# LB-Foraging environment args
	hunter_ids = args.hunter_ids
	prey_ids = args.prey_ids
	field_lengths = args.field_lengths
	max_steps = args.max_steps
	use_render = args.use_render
	require_catch = args.require_catch
	hunter_class = args.hunter_class
	
	hunters = []
	preys = []
	n_hunters = len(hunter_ids)
	n_preys = len(prey_ids)
	has_targets = (hunter_class != 0)
	for idx in range(n_hunters):
		hunters += [(hunter_ids[idx], hunter_class)]
	for idx in range(n_preys):
		preys += [(prey_ids[idx], 1)]
	
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	if not use_gpu:
		jax.config.update('jax_platform_name', 'cpu')
	
	# logger.info(gamma, initial_eps, final_eps, eps_decay, eps_type, warmup, learn_rate, target_learn_rate)
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

	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = (('train_pursuit_single_dqn_%dx%d-field_%d-hunters_%d-preys' % (field_size[0], field_size[1], n_hunters, n_preys)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / 'pursuit_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-hunters' % n_hunters) /
				  ('%d-preys' % n_preys) / now.strftime("%Y%m%d-%H%M%S"))
	
	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)
	
	logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
						level=logging.INFO, encoding='utf-8')
	logger = logging.getLogger('INFO')
	err_logger = logging.getLogger('ERROR')
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
	err_logger.addHandler(handler)
	Path.mkdir(model_path, parents=True, exist_ok=True)
	
	#####################
	## Training Models ##
	#####################
	logger.info('##########################')
	logger.info('Starting Pursuit DQN Train')
	logger.info('##########################')
	n_cycles = n_preys
	logger.info('Number of cycles: %d' % n_cycles)
	logger.info('Number of iterations per cycle: %d' % n_iterations)
	logger.info('Environment setup')
	env = PursuitEnv(hunters, preys, field_size, sight, require_catch, max_steps, use_layer_obs=True)
	logger.info('Setup multi-agent DQN')
	dqn_model = SingleModelMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.observation_space[0],
								 use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard,
								 tensorboard_details + ['%dh-%dp-%dc' % (n_hunters, n_preys, require_catch)])
	
	logger.info('Starting training')
	for cycle in range(n_cycles):
		logger.info('Cycle %d of %d' % (cycle+1, n_cycles))
		env.seed(RNG_SEED)
		sys.stdout.flush()
		if cycle == 0:
			cycle_init_eps = initial_eps
		else:
			cycle_init_eps = eps_cycle_schedule('exp', cycle, n_cycles, initial_eps, final_eps, cycle_decay)
		logger.info('Starting exploration: %d with decay of %f' % (cycle_init_eps, eps_decay))
		history = train_pursuit_dqn(dqn_model, env, n_iterations, max_steps * n_iterations, batch_size, learn_rate, target_update_rate, cycle_init_eps,
									final_eps, eps_type, RNG_SEED, logger, eps_decay, warmup, train_freq, target_freq, tensorboard_freq, use_render, cycle)
		
		if debug:
			logger.debug('Saving cycle iteration history')
			json_path = model_path / 'all_preys_history_centralized.json'
			with open(json_path, 'a') as json_file:
				json_file.write(json.dumps({('cycle_%d' % (cycle + 1)): history}))
		
	logger.info('Saving final model')
	dqn_model.save_model('all_preys', model_path, logger)
	sys.stdout.flush()

	####################
	## Testing Model ##
	####################
	env = PursuitEnv(hunters, preys, field_size, sight, require_catch, max_steps, use_layer_obs=True)
	env.seed(TEST_RNG_SEED)
	np.random.seed(TEST_RNG_SEED)
	rng_gen = np.random.default_rng(TEST_RNG_SEED)
	initial_target = prey_ids[rng_gen.integers(0, n_preys)]
	for hunter_id in env.hunter_ids:
		if isinstance(env.agents[hunter_id], TargetAgent):
			env.set_target(hunter_id, initial_target)
	failed_history = []
	tests_passed = 0
	for i in range(N_TESTS):
		env.reset_init_pos()
		obs, *_ = env.reset()
		logger.info('Test number %d' % (i + 1))
		logger.info('Prey locations: ' + ', '.join(['(%d, %d)' % env.agents[prey_id].pos for prey_id in env.prey_ids]))
		logger.info('Agent positions: ' + ', '.join(['(%d, %d)' % env.agents[hunter_id].pos for hunter_id in env.hunter_ids]))
		if has_targets:
			logger.info('Starting target: %s' % initial_target + ' (%d, %d)' % env.agents[initial_target].pos)
		obs, *_ = env.reset()
		epoch = 0
		agent_reward = [0, 0]
		test_history = []
		preys_alive = n_preys
		game_over = False
		finished = False
		timeout = False
		while not game_over:
			
			actions = []
			for h_idx in range(n_hunters):
				if dqn_model.agent_dqn.cnn_layer:
					q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params,
																   obs[h_idx].reshape((1, *obs[h_idx].shape)))[0]
				else:
					q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, obs[h_idx])
				action = q_values.argmax(axis=-1)
				action = jax.device_get(action)
				actions += [action]
			for prey_id in env.prey_ids:
				actions += [env.agents[prey_id].act(env)]
			actions = np.array(actions)
			next_obs, rewards, finished, timeout, infos = env.step(actions)
			test_history += [get_history_entry(env.make_array_obs(), actions, dqn_model.num_agents)]
			obs = next_obs
			
			if 0 < env.n_preys < preys_alive:
				nxt_target = env.prey_ids[rng_gen.integers(0, env.n_preys)]
				for hunter_id in env.hunter_ids:
					if isinstance(env.agents[hunter_id], TargetAgent):
						env.set_target(hunter_id, nxt_target)
				preys_alive -= 1
				if has_targets:
					logger.info('Next target: %s' % nxt_target)
			
			if finished or timeout:
				game_over = True
			
			sys.stdout.flush()
			epoch += 1
		
		if finished:
			tests_passed += 1
			logger.info('Test %d finished in success' % (i + 1))
			logger.info('Number of epochs: %d' % epoch)
			logger.info('Accumulated reward:\n\t- agent 1: %.2f\n\t- agent 2: %.2f' % (agent_reward[0], agent_reward[1]))
			logger.info('Average reward:\n\t- agent 1: %.2f\n\t- agent 2: %.2f' % (agent_reward[0] / epoch, agent_reward[1] / epoch))
		if timeout:
			failed_history += [test_history]
			logger.info('Test %d timed out' % (i + 1))
		
	logger.info('Passed %d tests out of %d' % (tests_passed, N_TESTS))
	logger.info('Failed tests history:')
	logger.info(failed_history)
	

if __name__ == '__main__':
	main()

