#! /usr/bin/env python

import argparse
import logging
import os
import sys
import jax
import numpy as np
import flax.linen as nn
import yaml
import csv
import json
import time
import signal
import multiprocessing as mp

from dl_algos.single_model_madqn import SingleModelMADQN
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv, CellEntity
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from itertools import product, combinations, permutations
from typing import List, Tuple, Dict
from termcolor import colored
from enum import IntEnum


RNG_SEED = 4072023
STATE_LEN = 8
ACTION_DIM = 6
MAX_EPOCH = 4000
PLAYER_POS = [(0, 0), (0, 7), (1, 3), (1, 6), (2, 1), (2, 5), (3, 3), (3, 4), (4, 2), (4, 7), (5, 0), (5, 5), (6, 2), (6, 4), (7, 0), (7, 4), (7, 7)]


class TestType(IntEnum):
	FULL = 0
	SINGLE_FOOD = 1
	CONFIGURATION = 2
	
	
class KillException(BaseException):
	
	def __init__(self, message="Got a signal to kill the process."):
		self.message = message
		super().__init__(self.message)
	
	def __str__(self):
		return self.message


def signal_handler(signum, frame):
	
	signame = signal.Signals(signum).name
	print('Signal handler called with signal %s' % signame)
	if signum == signal.SIGKILL.value:
		raise KillException()
	
	elif signum == signal.SIGINT.value:
		raise KeyboardInterrupt()


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def get_test_pos(env: LBForagingEnv) -> Tuple[List, List]:
	
	rows, cols = env.field.shape
	test_pos = []
	force_test_pos = []
	for p in product(range(rows), range(cols)):
		if env.field[p] == CellEntity.EMPTY:
			test_pos.append(p)
	
	for comb in combinations(test_pos, 2):
		for perm in permutations(comb, 2):
			force_test_pos.append(perm)
	
	return test_pos, force_test_pos
	

def test_configuration(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
					   n_layers: int, layer_sizes: List, buffer_size: int, gamma: float, use_gpu: bool, use_tensorboard: bool, tensorboard_details: List,
					   use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool, agent_ids: List, n_food_spawn: int, model_path: Path,
					   model_dirname: str, food_pos: List, obj_loc: Tuple, agent_pos: List, logger: logging.Logger) -> Dict:
	
	logger.info('#################################')
	logger.info('  Starting LB Foraging DQN Test  ')
	logger.info(' Testing for given configuration ')
	logger.info('#################################')
	logger.info('Environment setup')
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
							 use_encoding=True, agent_center=False, grid_observation=use_cnn)
	
	# DQN model loading
	results = {}
	loc = obj_loc
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	logger.info('Testing for location: %dx%d' % (loc[0], loc[1]))
	logger.info('Setup multi-agent DQN')
	obs_dims = [field_size[0], field_size[1], *([2] * (food_level + 1))] * n_foods + [field_size[0], field_size[1], *([2] * (player_level + 1))] * n_agents
	agents_dqn = SingleModelMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, MultiDiscrete(obs_dims),
								   use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard, tensorboard_details)
	logger.info(loc[0], loc[1], n_food_spawn, food_level)
	agents_dqn.load_model(('food_%dx%d' % (loc[0], loc[1])), model_path / ('%d-foods_%d-food-level' % (n_food_spawn, food_level)) / model_dirname,
						  logger, env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape))
	
	# Testing cycle
	logger.info('Starting testing cycles')
	avg_reward = 0
	avg_epochs = 0
	finished_epochs = 0
	n_epochs = []
	test_history = []
	initial_pos = []
	history = []
	game_over = False
	players_pos = agent_pos.copy()
	initial_pos.append([players_pos])
	food_spawn_pos = food_pos.copy()
	food_spawn_pos.remove(loc)
	env.obj_food = loc
	env.food_spawn_pos = food_spawn_pos
	env.spawn_food(n_food_spawn, food_level)
	env.spawn_players(player_level, players_pos)
	finished, timeout, epoch = False, False, 0
	try:
		logger.info('Cycle params:')
		logger.info('Agents positions:\t', [p.position for p in env.players])
		logger.info('Number of food spawn:\t%d' % n_food_spawn)
		logger.info('Objective food:\t(%d, %d)' % (loc[0], loc[1]))
		# logger.info('Testing field:')
		# logger.info(env.field)
		obs, *_ = env.reset()
		if use_render:
			env.render()
			input()
		
		epoch = 0
		acc_reward = 0
		logger.info('Initial observation:')
		logger.info(env.field)
		while not game_over:
			actions = []
			for a_idx in range(agents_dqn.num_agents):
				q_values = agents_dqn.agent_dqn.q_network.apply(agents_dqn.agent_dqn.online_state.params, obs[a_idx])
				action = q_values.argmax(axis=-1)
				action = jax.device_get(action)
				actions += [action]
			actions = np.array(actions)
			next_obs, rewards, finished, timeout, _ = env.step(actions)
			if use_render:
				env.render()
				time.sleep(0.1)
			obs_out, *_ = env.make_gym_obs()
			history += [get_history_entry(obs_out, actions, len(agent_ids))]
			acc_reward += sum(rewards) / n_agents
			obs = next_obs
			
			if finished or timeout or epoch >= max_steps:
				logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
				if finished and not timeout:
					avg_reward += acc_reward
					avg_epochs += epoch
					n_epochs += [epoch]
					finished_epochs += 1
				else:
					avg_epochs += max_steps
					n_epochs += [max_steps]
				game_over = True
				test_history += [max_steps]
			
			sys.stdout.flush()
			epoch += 1
		
		env.close()
		
		results[loc] = [finished_epochs, avg_reward, avg_epochs, n_epochs, test_history]
		logger.info('Loc: (%d, %d) results: ' % (loc[0], loc[1]), results[loc])
		logger.info('##########################################\n\n')
		
	except KeyboardInterrupt as ki:
		logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
		results[loc] = [finished_epochs, avg_reward, avg_epochs, n_epochs, test_history]
		logger.info('Keyboard interrupt caught %s\n Finishing testing cycle' % ki)
	
	return results
	

def test_full_scenario(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
					   n_layers: int, layer_sizes: List, buffer_size: int, gamma: float, use_gpu: bool, use_tensorboard: bool, tensorboard_details: List,
					   n_cycles: int, use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool, agent_ids: List, model_path: Path, model_dirname: str,
					   test_len: int, n_foods_spawn: int, logger: logging.Logger) -> Dict:
	
	logger.info('#################################')
	logger.info('  Starting LB Foraging DQN Test')
	logger.info('  Testing for full cycle')
	logger.info('#################################')
	logger.info('Environment setup')
	
	# DQN model loading
	results = {}
	for loc in [food_locs[0]]:
		results[loc] = run_loc_test_full(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes,
										 buffer_size, gamma, use_gpu, use_tensorboard, tensorboard_details, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn,
										 agent_ids, model_path, model_dirname, test_len, n_foods_spawn, loc, logger)
	
	return results

def test_number_foods(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
					  n_layers: int, layer_sizes: List, buffer_size: int, gamma: float, use_gpu: bool, use_tensorboard: bool, tensorboard_details: List,
					  n_cycles: int, use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool, agent_ids: List,
					  n_foods_spawn: int, model_path: Path, logger: logging.Logger) -> Dict:
	
	logger.info('#################################')
	logger.info('  Starting LB Foraging DQN Test')
	logger.info('  Testing for %d foods spawned' % n_foods_spawn)
	logger.info('#################################')
	
	# DQN model loading
	food_locs_test = food_locs
	results = {}
	for loc in food_locs_test:
		results[loc] = run_loc_test(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes, buffer_size,
									gamma, use_gpu, use_tensorboard, tensorboard_details, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids,
									n_foods_spawn, model_path, loc, logger)

	return results


def run_loc_test_full(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
					   n_layers: int, layer_sizes: List, buffer_size: int, gamma: float, use_gpu: bool, use_tensorboard: bool, tensorboard_details: List,
					   n_cycles: int, use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool, agent_ids: List, model_path: Path, model_dirname: str,
					   test_len: int, n_foods_spawn: int, loc: Tuple[int, int], logger: logging.Logger) -> Tuple:
	
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
							 use_encoding=True, agent_center=False, grid_observation=use_cnn)
	rnd_gen = np.random.default_rng(RNG_SEED)
	food_test_seqs = []
	locs_cp = food_locs.copy()
	locs_cp.pop(food_locs.index(loc))
	for _ in range(n_cycles):
		rnd_locs = rnd_gen.choice(range(len(locs_cp)), size=test_len - 1, replace=False)
		food_test_seqs.append([locs_cp[idx] for idx in rnd_locs])
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	rnd_gen = np.random.default_rng(RNG_SEED)
	avg_reward = 0
	avg_foods_caught = 0
	avg_epochs = 0
	finished_epochs = 0
	foods_caught = []
	n_epochs = []
	test_history = []
	initial_pos = []
	steps_food = []
	cycle = 0
	finished, timeout, epoch = False, False, 0
	obs_shape = env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape)
	try:
		for seq in food_test_seqs:
			remain_foods = n_foods_spawn
			test_seq = [loc] + seq.copy()
			logger.info('Test for sequence: ', test_seq)
			history = []
			game_over = False
			
			players_pos = rnd_gen.choice(PLAYER_POS, size=2, replace=False)
			initial_pos.append([players_pos])
			logger.info('Environment setup')
			env.obj_food = loc
			test_seq.pop(0)
			env.spawn_food(n_foods_spawn, food_level)
			env.spawn_players(player_level, players_pos)
			obs, *_ = env.reset()
			if use_render:
				env.render()
				input()
			
			logger.info('Setup multi-agent DQN')
			agents_dqn = SingleModelMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.observation_space[0],
										  use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard, tensorboard_details)
			agents_dqn.load_model(('food_%dx%d' % (loc[0], loc[1])), model_path / ('%d-foods_%d-food-level' % (remain_foods, food_level)) / model_dirname,
								  logger, obs_shape)
			
			epoch = 0
			acc_reward = 0
			logger.info('Initial observation:')
			logger.info(env.field)
			while not game_over:
				actions = []
				for a_idx in range(agents_dqn.num_agents):
					if agents_dqn.agent_dqn.cnn_layer:
						q_values = agents_dqn.agent_dqn.q_network.apply(agents_dqn.agent_dqn.online_state.params,
																		obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
					else:
						q_values = agents_dqn.agent_dqn.q_network.apply(agents_dqn.agent_dqn.online_state.params, obs[a_idx])
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
				next_obs, rewards, finished, timeout, _ = env.step(actions)
				if use_render:
					env.render()
					time.sleep(0.1)
				obs_out, *_ = env.make_gym_obs()
				history += [get_history_entry(obs_out, actions, len(agent_ids))]
				acc_reward += sum(rewards) / n_agents
				obs = next_obs
				
				remain_foods = env.count_foods()
				if finished:
					logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
					if remain_foods < 1:
						avg_reward += acc_reward / n_cycles
						avg_epochs += epoch / n_cycles
						avg_foods_caught += n_foods_spawn / n_cycles
						n_epochs += [epoch]
						finished_epochs += 1
						game_over = True
						test_history += [history]
						foods_caught += [n_foods_spawn]
					else:
						env.obj_food = test_seq.pop(0)
						agents_dqn.load_model(('food_%dx%d' % (env.obj_food[0], env.obj_food[1])),
											  model_path / ('%d-foods_%d-food-level' % (remain_foods, food_level)) / model_dirname, logger, obs_shape)
						steps_food.append(env.timestep)
						logger.info('New food observation:')
						logger.info(env.field)
					env.reset_timesteps()
				
				elif timeout or epoch >= MAX_EPOCH:
					logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
					if timeout:
						avg_epochs += max_steps / n_cycles
						n_epochs += [max_steps]
					else:
						avg_epochs += MAX_EPOCH / n_cycles
						n_epochs += [MAX_EPOCH]
					avg_foods_caught += (n_foods_spawn - remain_foods) / n_cycles
					game_over = True
					test_history += [history]
					foods_caught += [n_foods_spawn - remain_foods]
					steps_food.append(env.timestep)
				
				sys.stdout.flush()
				epoch += 1
			
			cycle += 1
			env.close()
			if use_render:
				env.close_render()
			
			logger.info('Environment hard reset')
			env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs)
			env.seed(RNG_SEED + cycle + 1)
			np.random.seed(RNG_SEED + cycle + 1)
		
		ratio_finished = finished_epochs / n_cycles
		logger.info('Loc: (%d, %d) results: ' % (loc[0], loc[1]),
			  [ratio_finished, avg_reward, avg_epochs, avg_foods_caught, n_epochs, test_history, foods_caught, steps_food])
		logger.info('##########################################\n\n')
		return ratio_finished, avg_reward, avg_epochs, avg_foods_caught, n_epochs, test_history, foods_caught, steps_food
	
	except KeyboardInterrupt as ki:
		logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
		logger.info('Keyboard interrupt caught %s\n Finishing testing cycle' % ki)
		return finished_epochs / n_cycles, avg_reward, avg_epochs, avg_foods_caught, n_epochs, test_history, foods_caught, steps_food


def run_loc_test(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
				 n_layers: int, layer_sizes: List, buffer_size: int, gamma: float, use_gpu: bool, use_tensorboard: bool, tensorboard_details: List,
				 n_cycles: int, use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool, agent_ids: List,
				 n_food_spawn: int, model_path: Path, loc: Tuple[int, int], logger: logging.Logger) -> Tuple:
	
	logger.info('Environment setup')
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
							 use_encoding=True, agent_center=False, grid_observation=use_cnn)
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	rnd_gen = np.random.default_rng(RNG_SEED)
	logger.info('Testing for location: %dx%d' % (loc[0], loc[1]))
	logger.info('Setup multi-agent DQN')
	agents_dqn = SingleModelMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.observation_space[0],
								   use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard, tensorboard_details)
	agents_dqn.load_model(('food_%dx%d' % (loc[0], loc[1])), model_path, logger,
						  env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape))
	
	# Testing cycle
	logger.info('Starting testing cycles')
	avg_reward = 0
	avg_epochs = 0
	finished_epochs = 0
	n_epochs = []
	test_history = []
	initial_pos = []
	env.obj_food = loc
	env.spawn_food(n_food_spawn, food_level)
	test_pos, force_test_pos = get_test_pos(env)
	n_force_cycles = len(force_test_pos)
	finished, timeout, epoch = False, False, 0
	try:
		for cycle in range(n_cycles):
			logger.info('##########################################')
			logger.info('Cycle %d out of %d' % (cycle + 1, n_cycles))
			history = []
			game_over = False
			if cycle < n_force_cycles:
				players_pos = force_test_pos[cycle]
			else:
				players_pos = rnd_gen.choice(test_pos, size=2, replace=False)
			initial_pos.append([players_pos])
			env.spawn_players(player_level, players_pos)
			logger.info('Cycle params:')
			logger.info('Agents positions:\t' + ', '.join(['(%d, %d)' % p.position for p in env.players]))
			logger.info('Number of food spawn:\t%d' % n_food_spawn)
			logger.info('Objective food:\t(%d, %d)' % (loc[0], loc[1]))
			obs, *_ = env.reset()
			if use_render:
				env.render()
				input()
			
			epoch = 0
			acc_reward = 0
			while not game_over:
				actions = []
				for a_idx in range(agents_dqn.num_agents):
					if agents_dqn.agent_dqn.cnn_layer:
						q_values = agents_dqn.agent_dqn.q_network.apply(agents_dqn.agent_dqn.online_state.params,
																		 obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
					else:
						q_values = agents_dqn.agent_dqn.q_network.apply(agents_dqn.agent_dqn.online_state.params, obs[a_idx])
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
				next_obs, rewards, finished, timeout, _ = env.step(actions)
				if use_render:
					env.render()
					time.sleep(0.1)
				history += [get_history_entry(env.make_obs_array(), actions, len(agent_ids))]
				acc_reward += sum(rewards) / n_agents
				obs = next_obs
				
				if finished or timeout or epoch >= max_steps:
					logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
					if finished and not timeout:
						avg_reward += acc_reward / n_cycles
						avg_epochs += epoch / n_cycles
						n_epochs += [epoch]
						finished_epochs += 1
					else:
						avg_epochs += max_steps / n_cycles
						n_epochs += [max_steps]
					game_over = True
					test_history += [history]
				
				sys.stdout.flush()
				epoch += 1
			
			env.close()
			
			logger.info('Environment hard reset')
			env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
									 use_encoding=True, agent_center=False, grid_observation=use_cnn)
			env.seed(RNG_SEED + cycle + 1)
			np.random.seed(RNG_SEED + cycle + 1)
			env.obj_food = loc
			env.spawn_food(n_food_spawn, food_level)
		
		ratio_finished = finished_epochs / n_cycles
		logger.info('Loc: (%d, %d) results:\n\t- completed: %f%%\n\t- average reward: %f\n\t- average number epochs: %f\n\t' %
					(loc[0], loc[1], ratio_finished * 100, avg_reward, avg_epochs))
		logger.info('##########################################\n\n')
		return ratio_finished, avg_reward, avg_epochs, n_epochs, test_history
	
	except KeyboardInterrupt as ki:
		logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
		logger.info('Keyboard interrupt caught %s\n Finishing testing cycle' % ki)
		return finished_epochs / n_cycles, avg_reward, avg_epochs, n_epochs, test_history


def write_results_full_file(data_dir: Path, filename: str, results: Dict, logger: logging.Logger) -> None:
	try:
		with open(data_dir / (filename + '.csv'), 'w') as results_file:
			headers = ['food_loc', 'ratio_finished', 'average_rewards', 'average_epochs', 'average_foods_caught']
			writer = csv.DictWriter(results_file, fieldnames=headers, delimiter=',', lineterminator='\n')
			writer.writeheader()
			for key in results.keys():
				row = {}
				for header, val in zip(headers, [key] + results[key]):
					row[header] = val
				writer.writerow(row)
		
		with open(data_dir / (filename + '_history.json'), 'w') as json_file:
			for key in results.keys():
				json_file.write(json.dumps({' '.join([str(x) for x in key]): results[key][-3]}))
		
		with open(data_dir / (filename + '_epochs_food.json'), 'w') as json_file:
			for key in results.keys():
				json_file.write(json.dumps({' '.join([str(x) for x in key]): [results[key][-4], results[key][-2], results[key][-1]]}))
	
	except IOError as e:
		logger.error("I/O error: " + str(e))


def write_results_file(data_dir: Path, filename: str, results: Dict, logger: logging.Logger) -> None:

	try:
		with open(data_dir / (filename + '.csv'), 'w') as results_file:
			headers = ['food_loc', 'ratio_finished', 'average_rewards', 'average_epochs']
			writer = csv.DictWriter(results_file, fieldnames=headers, delimiter=',', lineterminator='\n')
			writer.writeheader()
			for key in results.keys():
				row = {}
				for header, val in zip(headers, [key] + list(results[key])):
					row[header] = val
				writer.writerow(row)
				
		with open(data_dir / (filename + '.json'), 'w') as json_file:
			for key in results.keys():
				json_file.write(json.dumps({' '.join([str(x) for x in key]): results[key][-1]}))
	
	except IOError as e:
		logger.error("I/O error: " + str(e))
		

# noinspection DuplicatedCode
def main():
	parser = argparse.ArgumentParser(description='Test DQN model for Astro waste disposal game.')
	
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
	parser.add_argument('--agent-ids', dest='agent_ids', type=str, required=True, nargs='+', help='ID for each agent in the environment')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	
	# Testing params
	parser.add_argument('--cycles', dest='n_cycles', type=int, required=True,
						help='Number of testing cycles, each cycle spawns the field with a different number of food items.')
	parser.add_argument('--model-info', dest='model_info', type=str, nargs='+', help='List  with the info required to load the model to test: '
																					 '<model_dirname: str> <model_filename: str>')
	parser.add_argument('--test-mode', dest='test_mode', type=int, choices=[0, 1, 2], help='Type of test to run:\n\t0 - test on a full lb-foraging run, '
																						'picking all foods in environment\n\t1 - test lb-foraging models for a '
																						'specific number of spawned foods')
	parser.add_argument('--test-len', dest='test_len', type=int, required=False, help='Length of sequences for full sequence test')
	parser.add_argument('--parallel', dest='use_parallel', action='store_true', help='Flag triggering parallel execution of training cycles.')
	
	# Environment parameters
	parser.add_argument('--player-level', dest='player_level', type=int, required=True, help='Level of the agents collecting food')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--n-food', dest='n_foods', type=int, required=True, help='Number of food items in the field')
	parser.add_argument('--food-level', dest='food_level', type=int, required=True, help='Level of the food items')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag that signals the use of the field render while training')
	parser.add_argument('--n-foods-spawn', dest='n_foods_spawn', type=int, required=True, help='Number of foods to be spawned for training.')
	
	args = parser.parse_args()
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
	n_cycles = args.n_cycles
	agent_ids = args.agent_ids
	player_level = args.player_level
	field_lengths = args.field_lengths
	n_foods = args.n_foods
	food_level = args.food_level
	max_steps = args.max_steps
	use_render = args.use_render
	model_info = args.model_info
	n_foods_spawn = args.n_foods_spawn
	test_mode = args.test_mode
	test_len = args.test_len
	use_parallel = args.use_parallel
	
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	if not use_gpu:
		jax.config.update('jax_platform_name', 'cpu')
	
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
	
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	model_dirname = model_info[0]
	model_name = model_info[1]
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = config_params['food_locs'][dict_idx]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	signal.signal(signal.SIGINT, signal_handler)
	if test_mode == TestType.FULL:
		log_filename = ('test_lb_single_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level_%s-full' % (field_size[0], field_size[1], n_agents, n_foods_spawn,
																								 food_level, model_name))
		model_path = models_dir / 'lb_coop_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents)
		filename = 'test_lb_foraging_full'
		logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
							level=logging.INFO, encoding='utf-8')
		logger = logging.getLogger('INFO')
		err_logger = logging.getLogger('ERROR')
		handler = logging.StreamHandler(sys.stderr)
		handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
		err_logger.addHandler(handler)
		if use_parallel:
			t_pool = mp.Pool(int(0.75 * mp.cpu_count()))
			test_food_locs = food_locs
			pool_results = [t_pool.apply_async(run_loc_test_full, args=(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs,
																		n_layers, layer_sizes, buffer_size, gamma, use_gpu, use_tensorboard, tensorboard_details,
																		n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids, model_path,
																		model_dirname, test_len, n_foods_spawn, loc, logger)) for loc in test_food_locs]
			t_pool.close()
			results = {}
			for idx in range(len(pool_results)):
				loc = test_food_locs[idx]
				results[loc] = list(pool_results[idx].get())
			t_pool.join()
		else:
			results = test_full_scenario(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes, buffer_size,
										 gamma, use_gpu, use_tensorboard, tensorboard_details, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids,
										 model_path, model_dirname, test_len, n_foods_spawn, logger)
		write_results_full_file(data_dir, filename, results, logger)
	elif test_mode == TestType.SINGLE_FOOD:
		log_filename = ('test_lb_single_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level_%s' % (field_size[0], field_size[1], n_agents, n_foods_spawn,
																								 food_level, model_name))
		model_path = (models_dir / 'lb_coop_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
					  ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / model_dirname)
		filename = 'test_lb_foraging_%d_foods' % n_foods_spawn
		logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
							level=logging.INFO, encoding='utf-8')
		logger = logging.getLogger('INFO')
		err_logger = logging.getLogger('ERROR')
		handler = logging.StreamHandler(sys.stderr)
		handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
		err_logger.addHandler(handler)
		if use_parallel:
			t_pool = mp.Pool(int(0.75 * mp.cpu_count()))
			test_food_locs = food_locs
			pool_results = [t_pool.apply_async(run_loc_test, args=(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs,
																   n_layers, layer_sizes, buffer_size, gamma, use_gpu, use_tensorboard, tensorboard_details,
																   n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids, n_foods_spawn, model_path,
																   loc, logger)) for loc in test_food_locs]
			t_pool.close()
			results = {}
			for idx in range(len(pool_results)):
				loc = test_food_locs[idx]
				results[loc] = list(pool_results[idx].get())
			t_pool.join()
		else:
			results = test_number_foods(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes, buffer_size,
										gamma, use_gpu, use_tensorboard, tensorboard_details, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids,
										n_foods_spawn, model_path, logger)
		write_results_file(data_dir, filename + '_' + model_dirname, results, logger)
	elif test_mode == TestType.CONFIGURATION:
		log_filename = ('test_lb_single_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level_%s-config' % (field_size[0], field_size[1], n_agents, n_foods_spawn,
																										food_level, model_name))
		logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
							level=logging.INFO, encoding='utf-8')
		logger = logging.getLogger('INFO')
		err_logger = logging.getLogger('ERROR')
		handler = logging.StreamHandler(sys.stderr)
		handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
		err_logger.addHandler(handler)
		model_path = models_dir / 'lb_coop_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents)
		filename = 'test_lb_foraging_%d_foods_config' % n_foods_spawn
		food_pos = [(1, 7), (3, 0), (3, 1), (5, 4), (6, 6), (7, 1)]
		obj_loc = (7, 1)
		agents_pos = [(2, 6), (3, 7)]
		results = test_configuration(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes, buffer_size,
									 gamma, use_gpu, use_tensorboard, tensorboard_details, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids,
									 n_foods_spawn, model_path, model_dirname, food_pos, obj_loc, agents_pos)
		write_results_file(data_dir, filename, results, logger)
	else:
		print('[ARGS ERROR] Invalid testing mode, please review test mode.')
		return
	

if __name__ == '__main__':
	main()
