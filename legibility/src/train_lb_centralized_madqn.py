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

from dl_algos.single_model_madqn import CentralizedMADQN
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from pathlib import Path
from gymnasium.spaces import MultiBinary, Discrete
from itertools import product
from typing import List
from datetime import datetime


RNG_SEED = 13042023
TEST_RNG_SEED = 4072023
ACTION_DIM = 6


# noinspection DuplicatedCode
def number_food_combinations(max_foods: int, n_foods_spawn: int) -> int:
	return int(math.factorial(max_foods) / (math.factorial(n_foods_spawn) * math.factorial(max_foods - n_foods_spawn)))


def eps_cycle_schedule(cycle_nr: int, max_cycles: int, init_eps: float, final_eps: float, decay_rate: float) -> float:
	return max(init_eps - decay_rate ** ((max_cycles - 1) / cycle_nr), final_eps)


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def convert_joint_act(action: int, num_agents: int) -> List[int]:
	actions_map = list(product(range(ACTION_DIM), repeat=num_agents))
	return np.array(actions_map[action])


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
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	
	# Train parameters
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
		logger.info('[ARGS ERROR] Field size must either be composed of only 1 or 2 arguments; %d were given. Exiting program' % field_dims)
		return

	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = (('train_lb_coop_centralized_madqn_%dx%d-field_%d-agents_%d-foods_%d-food-level' % (field_size[0], field_size[1], n_agents,
																									   n_foods_spawn, food_level)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / 'lb_coop_central_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
				  ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / now.strftime("%Y%m%d-%H%M%S"))
	
	with open(data_dir / 'performances' / 'lb_foraging' / ('train_performances_centralized_%sa.yaml' % str(n_agents)), mode='r+',
			  encoding='utf-8') as train_file:
		train_performances = yaml.safe_load(train_file)
		field_idx = str(field_size[0]) + 'x' + str(field_size[1])
		food_idx = str(n_foods_spawn) + '-food'
		train_acc = train_performances[field_idx][food_idx]
	
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.safe_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = [tuple(x) for x in config_params['food_locs'][dict_idx]]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
		trained_keys = list(train_acc.keys())
		locs_train = []
		for loc in food_locs:
			key = '%s, %s' % (loc[0], loc[1])
			if key not in trained_keys or (key in trained_keys and train_acc[key] < 0.9):
				locs_train += [loc]
	
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
	logger.info('Environment setup')
	n_cycles = number_food_combinations(n_foods - 1, n_foods_spawn - 1) * 2
	logger.info(n_cycles)
	
	logger.info('Starting training for different food locations')
	for loc in locs_train:
		logger.info('Training for location: %dx%d' % (loc[0], loc[1]))
		logger.info('Environment setup')
		env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
								 use_encoding=True, agent_center=True, grid_observation=use_cnn)
		env.seed(RNG_SEED)
		env.set_objective(loc)
		logger.info('Setup multi-agent DQN')
		agent_action_space = Discrete(env.action_space[0].n ** n_agents)
		if isinstance(env.observation_space, MultiBinary):
			obs_space = env.observation_space
		else:
			obs_space = env.observation_space[0]
		madqn = CentralizedMADQN(n_agents, agent_action_space.n, n_layers, convert_joint_act, nn.relu, layer_sizes, buffer_size, gamma,
								 agent_action_space, obs_space, use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard,
								 tensorboard_details + ['%df-%dx%d' % (n_foods_spawn, loc[0], loc[1])])
		if restart_train:
			start_cycle = int(restart_info[2])
			logger.info('Load trained model')
			madqn.load_model(restart_info[1], model_path.parent.absolute() / restart_info[0])
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
			env.spawn_players([player_level] * n_agents)
			env.spawn_food(n_foods_spawn, food_level)
			madqn.replay_buffer.reset()
			logger.info('Cycle params:')
			logger.info('Number of food spawn:\t%d' % n_foods_spawn)
			logger.info('Food locations: ' + ', '.join(['(%d, %d)' % pos for pos in ([loc] + env.food_spawn_pos if n_foods_spawn < n_foods else food_locs)]))
			logger.info('Food objective: (%d, %d)' % env.obj_food)

			logger.info('Starting train')
			logger.info('Cycle starting epsilon: %f' % cycle_init_eps)
			history = madqn.train_dqn(env, n_iterations, max_steps * n_iterations, batch_size, learn_rate, target_update_rate, cycle_init_eps, final_eps,
											eps_type, RNG_SEED, logger, eps_decay, warmup, train_freq, target_freq, tensorboard_freq, use_render, cycle)

			# Reset params that determine how foods are spawn
			env.food_spawn_pos = None
			env.n_food_spawn = 0

			if debug:
				logger.info('Saving cycle iteration history')
				json_path = model_path / ('food_%dx%d_history_centralized.json' % (loc[0], loc[1]))
				with open(json_path, 'a') as json_file:
					json_file.write(json.dumps({('cycle_%d' % (cycle + 1)): history}))
	
				logger.info('Saving model after cycle %d' % (cycle + 1))
				Path.mkdir(model_path, parents=True, exist_ok=True)
				madqn.save_model(('food_%dx%d_cycle_%d' % (loc[0], loc[1], cycle + 1)), model_path, logger)
				sys.stdout.flush()

		logger.info('Saving final model')
		madqn.save_model(('food_%dx%d' % (loc[0], loc[1])), model_path, logger)
		sys.stdout.flush()

		logger.info('Testing for location: %dx%d' % (loc[0], loc[1]))
		env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
								 use_encoding=True, agent_center=True, grid_observation=use_cnn)
		env.seed(TEST_RNG_SEED)
		rng_gen = np.random.default_rng(TEST_RNG_SEED)
		np.random.seed(TEST_RNG_SEED)
		env.spawn_players(player_level)
		env.spawn_food(n_foods, food_level)
		obs, *_ = env.reset()
		epoch = 0
		history = []
		game_over = False
		while not game_over:

			dqn = madqn.madqn
			if dqn.cnn_layer:
				obs_shape = obs[0].shape
				joint_obs = obs.reshape((n_agents * obs_shape[0], *obs_shape[1:]))
			else:
				joint_obs = np.array([obs[i, 0, :] for i in range(len(obs))]).ravel()
				joint_obs = joint_obs.reshape((1, *joint_obs.shape))
			q_values = dqn.q_network.apply(dqn.online_state.params, joint_obs)
			joint_action = jax.device_get(q_values.argmax(axis=-1))
			actions = convert_joint_act(joint_action, n_agents)
			next_obs, rewards, finished, timeout, infos = env.step(actions)
			history += [get_history_entry(obs, actions, madqn.num_agents)]
			obs = next_obs
			
			if finished or timeout:
				game_over = True
				env.food_spawn_pos = None
				env.n_food_spawn = 0

			sys.stdout.flush()
			epoch += 1

		logger.info('Test history:')
		logger.info(history)
	

if __name__ == '__main__':
	main()
	