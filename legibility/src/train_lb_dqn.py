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

from dl_algos.multi_model_madqn import MultiAgentDQN
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from itertools import product
from typing import List
from datetime import datetime

RNG_SEED = 13042023
TEST_RNG_SEED = 4072023


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
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	parser.add_argument('--agent-ids', dest='agent_ids', type=str, required=True, nargs='+', help='ID for each agent in the environment')
	
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
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	agent_ids = args.agent_ids
	# Train args
	# n_cycles = args.n_cycles
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
		print('[ARGS ERROR] Field size must either be composed of only 1 or 2 arguments; %d were given. Exiting program' % field_dims)
		return

	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = (('train_lb_coop_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level' % (field_size[0], field_size[1], n_agents, n_foods_spawn, food_level)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / 'lb_coop_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
						 ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / now.strftime("%Y%m%d-%H%M%S"))
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1]) + '_food_locs'
		if dict_idx in config_params.keys():
			food_locs = config_params[dict_idx]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	sys.stdout = open(log_dir / (log_filename + '_log.txt'), 'a')
	sys.stderr = open(log_dir / (log_filename + '_err.txt'), 'w')
	Path.mkdir(model_path, parents=True, exist_ok=True)
	
	print('##############################')
	print('Starting LB Foraging DQN Train')
	print('##############################')
	print('Environment setup')
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs)
	n_cycles = number_food_combinations(n_foods - 1, n_foods_spawn - 1) * 2
	print(n_cycles)
	
	print('Starting training for different food locations')
	for loc in food_locs:
	# loc = food_locs[0]
	# loc = (5, 4)
		print('Training for location: %dx%d' % (loc[0], loc[1]))
		env.seed(RNG_SEED)
		rng_gen = np.random.default_rng(RNG_SEED)
		env.obj_food = loc
		print('Setup multi-agent DQN')
		obs_dims = [field_size[0], field_size[1], *([2] * (food_level + 1))] * n_foods + [field_size[0], field_size[1], *([2] * (player_level + 1))] * n_agents
		agents_dqns = MultiAgentDQN(n_agents, agent_ids, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, MultiDiscrete(obs_dims),
									use_gpu, dueling_dqn, use_ddqn, False, use_tensorboard, tensorboard_details)
		if restart_train:
			start_cycle = int(restart_info[2])
			print('Load trained model')
			agents_dqns.load_models(restart_info[1], model_path.parent.absolute() / restart_info[0])
			cycles_range = range(start_cycle, n_cycles)
			print('Restarting train from cycle %d' % start_cycle)
		else:
			cycles_range = range(n_cycles)
			print('Starting train')
		for cycle in cycles_range:
			print('Cycle %d of %d' % (cycle+1, n_cycles))
			if cycle == 0:
				cycle_init_eps = initial_eps
			else:
				cycle_init_eps = eps_cycle_schedule(cycle, n_cycles, initial_eps, final_eps, cycle_eps_decay)
			env.spawn_players(player_level)
			env.spawn_food(n_foods_spawn, food_level)
			print('Cycle params:')
			print('Number of food spawn:\t%d' % n_foods_spawn)
			print('Food locations: ', env.food_spawn_pos + [loc]  if n_foods_spawn < n_foods else food_locs)
			print('Food objective: ', env.obj_food)
		
			print('Starting train')
			sys.stdout.flush()
			print('Cycle EPS: %f' % cycle_init_eps)
			history = agents_dqns.train_dqns(env, n_iterations, max_steps * n_iterations, batch_size, learn_rate, target_update_rate, cycle_init_eps, final_eps,
											 eps_type, RNG_SEED, log_filename + '_log.txt', eps_decay, warmup, train_freq, target_freq, tensorboard_freq,
											 use_render, cycle)
			
			# Reset params that determine how foods are spawn
			env.food_spawn_pos = None
			env.n_food_spawn = 0
			
			print('Saving cycle iteration history')
			json_path = model_path / ('food_%dx%d_history.json' % (loc[0], loc[1]))
			with open(json_path, 'a') as json_file:
				json_file.write(json.dumps({('cycle_%d' % (cycle + 1)): history}))
		
			print('Saving model after cycle %d' % (cycle + 1))
			Path.mkdir(model_path, parents=True, exist_ok=True)
			agents_dqns.save_models(('food_%dx%d_cycle_%d' % (loc[0], loc[1], cycle + 1)), model_path)
			sys.stdout.flush()
		
		print('Saving final model')
		agents_dqns.save_models(('food_%dx%d' % (loc[0], loc[1])), model_path)
		sys.stdout.flush()
		
		print('Testing for location: %dx%d' % (loc[0], loc[1]))
		env.seed(TEST_RNG_SEED)
		rng_gen = np.random.default_rng(TEST_RNG_SEED)
		np.random.seed(TEST_RNG_SEED)
		env.spawn_players(player_level)
		env.spawn_food(n_foods, food_level)
		print([p.position for p in env.players])
		obs, _, _, _ = env.reset()
		epoch = 0
		history = []
		game_over = False
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
			next_obs, rewards, finished, infos = env.step(actions)
			history += [get_history_entry(obs, actions, len(agent_ids))]
			obs = next_obs
			
			if all(finished) or epoch >= 500:
				game_over = True
			
			sys.stdout.flush()
			epoch += 1
		
		print('Test history:')
		print(history)
	

if __name__ == '__main__':
	main()
	