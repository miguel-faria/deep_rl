#! /usr/bin/env python

import os
import sys
import argparse
import numpy as np
import flax.linen as nn
import yaml
import jax
import json
import math
import logging

from dl_algos.single_model_madqn import SingleModelMADQN
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from pathlib import Path
from itertools import product
from typing import List
from datetime import datetime


RNG_SEED = 13042023
TEST_RNG_SEED = 4072023
N_TESTS = 100

def number_food_combinations(max_foods: int, n_foods_spawn: int) -> int:
	return max(1, int(math.factorial(max_foods) / (math.factorial(n_foods_spawn) * math.factorial(max_foods - n_foods_spawn)) * 0.75))


def eps_cycle_schedule(cycle_nr: int, max_cycles: int, init_eps: float, final_eps: float, decay_rate: float) -> float:
	return max(init_eps - decay_rate ** ((max_cycles - 1) / cycle_nr), final_eps)


def cycle_foods_spawn(max_foods_spawn: int, cycle: int, rng_gen: np.random.Generator) -> int:
	if cycle < 2:
		return max_foods_spawn
	else:
		return int(rng_gen.integers(1, max_foods_spawn + 1))


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


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
	# parser.add_argument('--cycles', dest='n_cycles', type=int, required=True,
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
	
	# Environment parameters
	parser.add_argument('--player-level', dest='player_level', type=int, required=True, help='Level of the agents collecting food')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--n-food', dest='n_foods', type=int, required=True, help='Number of food items in the field')
	parser.add_argument('--food-level', dest='food_level', type=int, required=True, help='Level of the food items')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag that signals the use of the field render while training')
	parser.add_argument('--n-foods-spawn', dest='n_foods_spawn', type=int, required=True, help='Number of foods to be spawned for training.')
	
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
	cycle_eps_decay = args.cycle_eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	restart_train = args.restart_train
	restart_info = args.restart_info
	debug = args.debug
	
	# LB-Foraging environment args
	player_level = args.player_level
	field_lengths = args.field_lengths
	n_foods = args.n_foods
	food_level = args.food_level
	max_steps = args.max_steps
	use_render = args.use_render
	n_foods_spawn = args.n_foods_spawn
	
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
		logging.error('[ARGS ERROR] Field size must either be composed of only 1 or 2 arguments; %d were given. Exiting program' % field_dims)
		return

	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = (('train_lb_coop_central_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level' % (field_size[0], field_size[1], n_agents,
																								 n_foods_spawn, food_level)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / 'lb_coop_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
				  ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / now.strftime("%Y%m%d-%H%M%S"))
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = config_params['food_locs'][dict_idx]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)
	
	logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
						level=logging.INFO)
	logger = logging.getLogger('INFO')
	err_logger = logging.getLogger('ERROR')
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
	err_logger.addHandler(handler)
	Path.mkdir(model_path, parents=True, exist_ok=True)
	
	logger.info('##############################')
	logger.info('Starting LB Foraging DQN Train')
	logger.info('##############################')
	n_cycles = number_food_combinations(n_foods - 1, n_foods_spawn - 1)
	logger.info('Number of cycles: %d' % n_cycles)
	
	####################
	## Training Model ##
	####################
	logger.info('Starting training for different food locations')
	for loc in food_locs:
		logger.info('Training for location: %dx%d' % (loc[0], loc[1]))
		logger.info('Environment setup')
		env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
								 use_encoding=True, agent_center=False, grid_observation=use_cnn)
		env.seed(RNG_SEED)
		env.set_objective(loc)
		logger.info('Setup multi-agent DQN')
		central_madqn = SingleModelMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.observation_space[0],
										 use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard,
										 tensorboard_details + ['%df-%dx%d' % (n_foods_spawn, loc[0], loc[1])])
		if restart_train:
			start_cycle = int(restart_info[2])
			logger.info('Load trained model')
			central_madqn.load_model(restart_info[1], model_path.parent.absolute() / restart_info[0], logger,
									 env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape))
			cycles_range = range(start_cycle, n_cycles)
			logger.info('Restarting train from cycle %d' % start_cycle)
		else:
			cycles_range = range(n_cycles)
			logger.info('Starting train')
		
		for cycle in cycles_range:
			logger.info('Cycle %d of %d' % (cycle+1, n_cycles))
			if cycle == 0:
				cycle_init_eps = initial_eps
			else:
				cycle_init_eps = eps_cycle_schedule(cycle, n_cycles, initial_eps, final_eps, cycle_eps_decay)
			env.spawn_players(player_level)
			env.spawn_food(n_foods_spawn, food_level)
			logger.info('Cycle params:')
			logger.info('Number of food spawn:\t%d' % n_foods_spawn)
			logger.info('Food locations: ' + ', '.join(['(%d, %d)' % pos for pos in ([loc] + env.food_spawn_pos if n_foods_spawn < n_foods else food_locs)]))
			logger.info('Food objective: (%d, %d)' % env.obj_food)
		
			logger.info('Starting train')
			logger.info('Cycle starting epsilon: %f' % cycle_init_eps)
			history = central_madqn.train_dqn(env, n_iterations, max_steps * n_iterations, batch_size, learn_rate, target_update_rate, cycle_init_eps, final_eps,
											eps_type, RNG_SEED, logger, eps_decay, warmup, train_freq, target_freq, tensorboard_freq,
											use_render, cycle, greedy_action=False)
			
			# Reset params that determine how foods are spawn
			env.food_spawn_pos = None
			env.food_spawn = 0
			
			if debug:
				logger.info('Saving cycle iteration history')
				json_path = model_path / ('food_%dx%d_history_centralized.json' % (loc[0], loc[1]))
				with open(json_path, 'a') as json_file:
					json_file.write(json.dumps({('cycle_%d' % (cycle + 1)): history}))
			
				logger.info('Saving model after cycle %d' % (cycle + 1))
				Path.mkdir(model_path, parents=True, exist_ok=True)
				central_madqn.save_model(('food_%dx%d_cycle_%d' % (loc[0], loc[1], cycle + 1)), model_path, logger)
		
		logger.info('Saving final model')
		central_madqn.save_model(('food_%dx%d' % (loc[0], loc[1])), model_path, logger)
		
		####################
		## Testing Model ##
		####################
		logger.info('Testing for location: %dx%d' % (loc[0], loc[1]))
		env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, TEST_RNG_SEED, food_locs,
								 use_encoding=True, agent_center=False, grid_observation=use_cnn)
		failed_history = []
		tests_passed = 0
		env.seed(TEST_RNG_SEED)
		np.random.seed(TEST_RNG_SEED)
		rng_gen = np.random.default_rng(TEST_RNG_SEED)
		for i in range(N_TESTS):
			env.set_objective(loc)
			env.spawn_players(player_level)
			env.spawn_food(n_foods_spawn, food_level)
			logger.info('Test number %d' % (i + 1))
			logger.info('Number of food spawn:\t%d' % n_foods_spawn)
			logger.info('Food locations: ' + ', '.join(['(%d, %d)' % pos for pos in ([loc] + env.food_spawn_pos if n_foods_spawn < n_foods else food_locs)]))
			logger.info('Agent positions: ' + ', '.join(['(%d, %d)' % p.position for p in env.players]))
			obs, *_ = env.reset()
			epoch = 0
			agent_reward = [0, 0]
			test_history = []
			game_over = False
			finished = False
			timeout = False
			while not game_over:
				
				actions = []
				for a_idx in range(central_madqn.num_agents):
					dqn = central_madqn.agent_dqn
					if use_cnn:
						q_values = dqn.q_network.apply(dqn.online_state.params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
					else:
						q_values = dqn.q_network.apply(dqn.online_state.params, obs[a_idx])
					pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
					pol = pol / pol.sum()
					action = rng_gen.choice(range(env.action_space[0].n), p=pol)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
				next_obs, rewards, finished, timeout, infos = env.step(actions)
				agent_reward[0] += rewards[0]
				agent_reward[1] += rewards[1]
				test_history += [get_history_entry(env.make_obs_array(), actions, central_madqn.num_agents)]
				obs = next_obs
				
				if finished or timeout:
					game_over = True
					env.food_spawn_pos = None
					env.food_spawn = 0
				
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
		
		logger.info('Passed %d tests out of %d for location %dx%d' % (tests_passed, N_TESTS, loc[0], loc[1]))
		logger.info('Failed tests history:')
		logger.info(failed_history)
	

if __name__ == '__main__':
	main()
	