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
import threading

from dl_algos.single_model_madqn import SingleModelMADQN
from dl_algos.dqn import DQNetwork
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv, CellEntity, Action
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from pathlib import Path
from gymnasium.spaces import MultiDiscrete, MultiBinary
from itertools import product, combinations, permutations
from typing import List, Tuple, Dict
from scipy.stats import sem
from enum import IntEnum
from datetime import datetime


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


def input_callback(env: FoodCOOPLBForaging, stop_flag: threading.Event):
	try:
		while not stop_flag.is_set():
			command = input('Interactive commands:\n\trender - display renderization of the interaction\n\tstop_render - stops the renderization\nCommand: ')
			if command == 'render':
				env.use_render = True
			elif command == 'stop_render':
				if env.use_render:
					env.close_render()
					env.use_render = False
	
	except KeyboardInterrupt as ki:
		return


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
					   num_layers: int, layer_sizes: List, gamma: float, use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool, agent_ids: List,
					   n_food_spawn: int, model_path: Path, model_dirname: str, food_pos: List, obj_loc: Tuple, agent_pos: List, logger: logging.Logger,
					   cnn_properties: List = None) -> Dict:
	
	logger.info('#################################')
	logger.info('  Starting LB Foraging DQN Test  ')
	logger.info(' Testing for given configuration ')
	logger.info('#################################')
	logger.info('Environment setup')
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
							 use_encoding=True, agent_center=False, grid_observation=use_cnn)
	n_actions = env.action_space[0].n
	stop_thread = False
	command_thread = threading.Thread(target=input_callback, args=(env, stop_thread))
	command_thread.start()
	# DQN model loading
	results = {}
	loc = obj_loc
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	logger.info('Testing for location: %dx%d' % (loc[0], loc[1]))
	logger.info('Setup multi-agent DQN')
	obs_shape = env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape)
	agents_dqn = DQNetwork(n_actions, num_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties=cnn_properties)
	logger.info(loc[0], loc[1], n_food_spawn, food_level)
	agents_dqn.load_model(('food_%dx%d' % (loc[0], loc[1])), model_path / ('%d-foods_%d-food-level' % (n_food_spawn, food_level)) / model_dirname,
						  logger, obs_shape)
	
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
	env.spawn_players([player_level] * n_agents, players_pos)
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
			for a_idx in range(n_agents):
				if agents_dqn.cnn_layer:
					q_values = agents_dqn.q_network.apply(agents_dqn.online_state.params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
				else:
					q_values = agents_dqn.q_network.apply(agents_dqn.online_state.params, obs[a_idx])
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
	
	stop_thread = True
	return results
	

def test_full_scenario(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
					   num_layers: int, layer_sizes: List, gamma: float, n_cycles: int, use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool,
					   agent_ids: List, model_path: Path, model_dirname: str, test_len: int, n_foods_spawn: int, logger: logging.Logger, debug: bool = False,
					   cnn_properties: List = None) -> Dict:
	
	logger.info('#################################')
	logger.info('  Starting LB Foraging DQN Test')
	logger.info('  Testing for full cycle')
	logger.info('#################################')
	logger.info('Environment setup')
	
	# DQN model loading
	results = {}
	for loc in food_locs:
		key = ', '.join([str(pos) for pos in loc])
		results[key] = run_loc_test_full(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, num_layers, layer_sizes,
										 gamma, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids, model_path, model_dirname, test_len,
										 n_foods_spawn, loc, logger, debug, cnn_properties)

	return results

def test_number_foods(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
					  num_layers: int, layer_sizes: List, gamma: float, n_cycles: int, use_render: bool, dueling_dqn: bool, use_ddqn: bool, use_cnn: bool,
					  agent_ids: List, n_foods_spawn: int, model_path: Path, logger: logging.Logger, debug: bool = False, cnn_properties: List = None) -> Dict:
	
	logger.info('#################################')
	logger.info('  Starting LB Foraging DQN Test')
	logger.info('  Testing for %d foods spawned' % n_foods_spawn)
	logger.info('#################################')
	
	# DQN model loading
	food_locs_test = food_locs
	results = {}
	for loc in food_locs_test:
		key = ', '.join([str(pos) for pos in loc])
		results[key] = run_loc_test(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, num_layers, layer_sizes,
									gamma, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids, n_foods_spawn, model_path, loc, logger,
									debug, cnn_properties=cnn_properties)

	return results


def run_loc_test_full(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
					  num_layers: int, layer_sizes: List, gamma: float, n_cycles: int, use_render: bool, dueling_dqn: bool,
					  use_ddqn: bool, use_cnn: bool, agent_ids: List, model_path: Path, model_dirname: str, test_len: int, n_foods_spawn: int,
					  loc: Tuple[int, int], logger: logging.Logger, debug: bool = False, cnn_properties: List = None) -> Tuple:
	
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
							 use_encoding=True, agent_center=True, grid_observation=use_cnn)
	n_actions = env.action_space[0].n
	stop_thread = threading.Event()
	command_thread = threading.Thread(target=input_callback, args=(env, stop_thread))
	command_thread.start()
	rn_gen = np.random.default_rng(RNG_SEED)
	food_test_seqs = []
	locs_cp = food_locs.copy()
	locs_cp.pop(food_locs.index(loc))
	for _ in range(n_cycles):
		rnd_locs = rn_gen.choice(range(len(locs_cp)), size=test_len - 1, replace=False)
		food_test_seqs.append([locs_cp[idx] for idx in rnd_locs])
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	rn_gen = np.random.default_rng(RNG_SEED)
	avg_reward = 0
	avg_foods_caught = 0
	avg_epochs = 0
	finished_epochs = 0
	foods_caught = []
	n_epochs = []
	run_rewards = []
	test_history = []
	initial_pos = []
	steps_food = []
	cycle = 0
	finished, timeout, epoch = False, False, 0
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	obs_shape = obs_space.shape if not use_cnn else (1, *obs_space.shape)
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	seq_nr = 0
	try:
		for seq in food_test_seqs:
			remain_foods = n_foods_spawn
			test_seq = [loc] + seq.copy()
			logger.info('Test for sequence %d out of %d: %s' % ((seq_nr + 1), n_cycles, ', '.join(['(%d, %d)' % (pos[0], pos[1]) for pos in test_seq])))
			history = []
			game_over = False
			
			players_pos = rn_gen.choice(PLAYER_POS, size=2, replace=False)
			initial_pos.append([players_pos])
			logger.info('Environment setup')
			env.obj_food = loc
			test_seq.pop(0)
			env.spawn_food(n_foods_spawn, food_level)
			env.spawn_players([player_level] * n_agents, players_pos)
			obs, *_ = env.reset()
			if use_render:
				env.render()
				time.sleep(0.1)
			
			logger.info('Setup multi-agent DQN')
			agents_dqn = DQNetwork(n_actions, num_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties=cnn_properties)
			agents_dqn.load_model(('food_%dx%d_single_model' % (loc[0], loc[1])), model_path / ('%d-foods_%d-food-level' % (remain_foods, food_level)) / model_dirname,
								  logger, obs_shape)
			
			epoch = 0
			acc_reward = 0
			logger.info('Initial Information')
			logger.info('Objective food:\t(%d, %d)' % (loc[0], loc[1]))
			logger.info(env.get_full_env_log())
			while not game_over:
				if debug:
					logger.info('Player\'s observations:')
					for idx in range(env.n_players):
						p_obs = obs[idx]
						logger.info('Player %s observation:\n%s' % (env.players[idx].player_id, '\n'.join([str(layer) for layer in p_obs])))
				actions = []
				for a_idx in range(n_agents):
					if agents_dqn.cnn_layer:
						q_values = agents_dqn.q_network.apply(agents_dqn.online_state.params,obs[a_idx].reshape((1, *cnn_shape)))[0]
					else:
						q_values = agents_dqn.q_network.apply(agents_dqn.online_state.params, obs[a_idx])
					pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
					pol = pol / pol.sum()
					action = rn_gen.choice(range(env.action_space[0].n), p=pol)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
				next_obs, rewards, finished, timeout, _ = env.step(actions)
				logger.info('Action: %s\n' % str([Action(act).name for act in actions]))
				logger.info(env.get_full_env_log())
				if use_render:
					env.render()
					time.sleep(0.1)
				history += [get_history_entry(env.make_obs_array(), actions, len(agent_ids))]
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
						run_rewards += [acc_reward / n_cycles]
					else:
						env.obj_food = test_seq.pop(0)
						agents_dqn.load_model(('food_%dx%d' % (env.obj_food[0], env.obj_food[1])),
											  model_path / ('%d-foods_%d-food-level' % (remain_foods, food_level)) / model_dirname, logger, obs_shape)
						steps_food.append(env.timestep)
						logger.info('New food observation:')
						logger.info(env.field)
						logger.info('Agents positions:\t' + ', '.join(['(%d, %d)' % p.position for p in env.players]))
						logger.info('Foods remaining: %d' % env.count_foods())
						logger.info('Food locations: ' + ', '.join(['(%d, %d)' % food.position for food in env.foods if not food.picked]))
						logger.info('Objective food:\t(%d, %d)' % env.obj_food)
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
					run_rewards += [acc_reward / n_cycles]
					steps_food.append(env.timestep)
				
				sys.stdout.flush()
				epoch += 1
			
			cycle += 1
			env.close()
			if use_render:
				env.close_render()
			
			logger.info('Environment hard reset')
			env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED + cycle + 1, food_locs,
									 use_encoding=True, agent_center=False, grid_observation=use_cnn)
			env.seed(RNG_SEED + cycle + 1)
			np.random.seed(RNG_SEED + cycle + 1)
			seq_nr += 1
		
		ratio_finished = finished_epochs / n_cycles
		epochs_sem = sem(n_epochs)
		foods_caught_sem = sem(foods_caught)
		rewards_sem = sem(run_rewards)
		logger.info('Loc: (%d, %d) results:\n\t- completed: %f%%\n\t- average reward: %f +/- %f\n\t- average number epochs: %f +/- %f\n\t'
					'- average number of foods caught: %f +/- %f\n' %
					(loc[0], loc[1], ratio_finished * 100, avg_reward, rewards_sem, avg_epochs, epochs_sem, avg_foods_caught, foods_caught_sem))
		logger.info('##########################################\n\n')
		stop_thread.set()
		env.close()
		command_thread.join()
		return (ratio_finished, avg_reward, rewards_sem, avg_epochs, epochs_sem, avg_foods_caught, foods_caught_sem, n_epochs, test_history,
				foods_caught, steps_food)
	
	except KeyboardInterrupt as ki:
		logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
		logger.info('Keyboard interrupt caught %s\n Finishing testing cycle' % ki)
		epochs_sem = sem(n_epochs)
		foods_caught_sem = sem(foods_caught)
		rewards_sem = sem(run_rewards)
		stop_thread.set()
		env.close()
		command_thread.join()
		return (finished_epochs / n_cycles, avg_reward, rewards_sem, avg_epochs, epochs_sem, avg_foods_caught, foods_caught_sem, n_epochs, test_history,
				foods_caught, steps_food)


def run_loc_test(n_agents: int, player_level: int, field_size: Tuple[int, int], n_foods: int, sight: int, max_steps: int, food_level: int, food_locs: List,
				 num_layers: int, layer_sizes: List, gamma: float, n_cycles: int, use_render: bool, dueling_dqn: bool,
				 use_ddqn: bool, use_cnn: bool, agent_ids: List, n_foods_spawn: int, model_path: Path, loc: Tuple[int, int], logger: logging.Logger,
				 debug: bool = False, cnn_properties: List = None) -> Tuple:
	
	logger.info('Environment setup')
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
							 use_encoding=True, agent_center=True, grid_observation=use_cnn, use_render=use_render)
	n_actions = env.action_space[0].n
	stop_thread = threading.Event()
	command_thread = threading.Thread(target=input_callback, args=(env, stop_thread))
	command_thread.start()
	env.seed(RNG_SEED)
	np.random.seed(RNG_SEED)
	rn_gen = np.random.default_rng(RNG_SEED)
	logger.info('Testing model %s for location: %dx%d' % (model_path.name, loc[0], loc[1]))
	logger.info('Setup multi-agent DQN')
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	obs_shape = obs_space.shape if not use_cnn else (1, *obs_space.shape)
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	# agents_dqn = SingleModelMADQN(n_agents, agent_action_space.n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, action_space,
	# 							  obs_space, use_gpu, dueling_dqn, use_ddqn, use_vdn, use_cnn, False, False)
	agents_dqn = DQNetwork(n_actions, num_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties=cnn_properties)
	agents_dqn.load_model(('food_%dx%d_single_model' % (loc[0], loc[1])), model_path, logger, obs_shape)
	
	# Testing cycle
	logger.info('Starting testing cycles')
	avg_reward = 0
	avg_epochs = 0
	finished_epochs = 0
	n_epochs = []
	test_history = []
	initial_pos = []
	test_pos, force_test_pos = get_test_pos(env)
	n_force_cycles = len(force_test_pos)
	finished, timeout, epoch = False, False, 0
	try:
		for cycle in range(n_cycles):
			logger.info('##########################################')
			logger.info('Cycle %d out of %d' % (cycle + 1, n_cycles))
			history = []
			game_over = False
			# if cycle < n_force_cycles:
			# 	players_pos = force_test_pos[cycle]
			# else:
			players_pos = rn_gen.choice(test_pos, size=2, replace=False)
			initial_pos.append([players_pos])
			env.set_objective(loc)
			env.spawn_players([player_level] * n_agents, players_pos)
			env.spawn_food(n_foods_spawn, food_level)
			obs, *_ = env.reset()
			logger.info('Cycle params:')
			logger.info(env.get_full_env_log())
			if env.use_render:
				env.render()
			else:
				env.close_render()
			
			epoch = 0
			acc_reward = 0
			while not game_over:
				if debug:
					logger.info('Player\'s observations:')
					for idx in range(env.n_players):
						p_obs = obs[idx]
						logger.info('Player %s observation:\n%s' % (env.players[idx].player_id, '\n'.join([str(layer) for layer in p_obs])))
				actions = []
				for a_idx in range(n_agents):
					if agents_dqn.cnn_layer:
						q_values = agents_dqn.q_network.apply(agents_dqn.online_state.params, obs[a_idx].reshape(1, *cnn_shape))[0]
					else:
						q_values = agents_dqn.q_network.apply(agents_dqn.online_state.params, obs[a_idx])
					pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
					pol = pol / pol.sum()
					action = rn_gen.choice(range(env.action_space[0].n), p=pol)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
				next_obs, rewards, finished, timeout, _ = env.step(actions)
				if env.use_render:
					env.render()
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
					env.food_spawn_pos = None
					env.n_food_spawn = 0
					test_history += [history]
				
				sys.stdout.flush()
				epoch += 1
			
			logger.info('Environment hard reset')
			np.random.seed(RNG_SEED + cycle + 1)
			env.seed(RNG_SEED + cycle + 1)
			rn_gen = np.random.default_rng(RNG_SEED + cycle + 1)
		
		ratio_finished = finished_epochs / n_cycles
		logger.info('Loc: (%d, %d) results:\n\t- completed: %f%%\n\t- average reward: %f\n\t- average number epochs: %f\n\t' %
					(loc[0], loc[1], ratio_finished * 100, avg_reward, avg_epochs))
		logger.info('##########################################\n\n')
		
		stop_thread.set()
		env.close()
		command_thread.join()
		return ratio_finished, avg_reward, avg_epochs, n_epochs, test_history
	
	except KeyboardInterrupt as ki:
		logger.info('Finished? %r\nTimeout? %r\nEpoch: %d' % (finished, timeout, epoch))
		logger.info('Keyboard interrupt caught %s\n Finishing testing cycle' % ki)
		stop_thread.set()
		env.close()
		command_thread.join()
		return finished_epochs / n_cycles, avg_reward, avg_epochs, n_epochs, test_history


def write_results_full_file(data_dir: Path, filename: str, results: Dict, logger: logging.Logger) -> None:
	try:
		with open(data_dir / (filename + '.csv'), 'w') as results_file:
			headers = ['food_loc', 'ratio_finished', 'average_rewards', 'rewards_error', 'average_epochs',
					   'epochs_error', 'average_foods_caught', 'food_caught_error']
			writer = csv.DictWriter(results_file, fieldnames=headers, delimiter=',', lineterminator='\n')
			writer.writeheader()
			for key in results.keys():
				row = {}
				for header, val in zip(headers, [key] + list(results[key])):
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
	parser.add_argument('--architecture', dest='architecture', type=str, required=True, help='DQN architecture to use from the architectures yaml')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
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
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	
	args = parser.parse_args()
	
	# DQN parameters
	n_agents = args.n_agents
	architecture = args.architecture
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_cnn = args.use_cnn
	use_vdn = args.use_vdn
	
	# Environment parameters
	agent_ids = args.agent_ids
	player_level = args.player_level
	field_lengths = args.field_lengths
	debug = args.debug
	n_foods_spawn = args.n_foods_spawn
	use_render = args.use_render
	n_foods = args.n_foods
	food_level = args.food_level
	max_steps = args.max_steps
	
	# Testing parameters
	n_cycles = args.n_cycles
	model_info = args.model_info
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
	
	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	model_dirname = model_info[0]
	model_name = model_info[1]
	
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.safe_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = [tuple(x) for x in config_params['food_locs'][dict_idx]]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	with open(data_dir / 'configs' / 'q_network_architectures.yaml') as architecture_file:
		arch_data = yaml.safe_load(architecture_file)
		if architecture in arch_data.keys():
			n_layers = arch_data[architecture]['n_layers']
			layer_sizes = arch_data[architecture]['layer_sizes']
			n_conv_layers = arch_data[architecture]['n_cnn_layers']
			cnn_size = arch_data[architecture]['cnn_size']
			cnn_kernel = [tuple(elem) for elem in arch_data[architecture]['cnn_kernel']]
			pool_window = [tuple(elem) for elem in arch_data[architecture]['pool_window']]
			cnn_properties = [n_conv_layers, cnn_size, cnn_kernel, pool_window]
	
	signal.signal(signal.SIGINT, signal_handler)
	if test_mode == TestType.FULL:
		log_filename = ('test_lb_single_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level_%s-full' % (field_size[0], field_size[1], n_agents, n_foods_spawn,
																								 food_level, model_name) + '_' + now.strftime("%Y%m%d-%H%M%S"))
		model_path = (models_dir / ('lb_coop_single%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
					  ('%d-agents' % n_agents))
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
																		n_layers, layer_sizes, gamma, n_cycles, use_render, dueling_dqn,
																		use_ddqn, use_cnn, agent_ids, model_path, model_dirname, test_len, n_foods_spawn, loc,
																		logger, debug, cnn_properties)) for loc in test_food_locs]
			t_pool.close()
			results = {}
			for idx in range(len(pool_results)):
				loc = test_food_locs[idx]
				results[loc] = list(pool_results[idx].get())
			t_pool.join()
		else:
			results = test_full_scenario(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes,
										 gamma, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids, model_path, model_dirname, test_len,
										 n_foods_spawn, logger, debug, cnn_properties)
		write_results_full_file(data_dir / 'performances' / 'lb_foraging', filename, results, logger)
	elif test_mode == TestType.SINGLE_FOOD:
		log_filename = ('test_lb_single_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level_%s' % (field_size[0], field_size[1], n_agents, n_foods_spawn,
																								 food_level, model_dirname) + '_' + now.strftime("%Y%m%d-%H%M%S"))
		model_path = (models_dir / ('lb_coop_single%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
					  ('%d-agents' % n_agents) / ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / model_dirname)
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
																   n_layers, layer_sizes, gamma, n_cycles, use_render, dueling_dqn,
																   use_ddqn, use_cnn, agent_ids, n_foods_spawn, model_path,
																   loc, logger, debug, cnn_properties)) for loc in test_food_locs]
			t_pool.close()
			results = {}
			for idx in range(len(pool_results)):
				loc = test_food_locs[idx]
				results[loc] = list(pool_results[idx].get())
			t_pool.join()
		else:
			results = test_number_foods(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes,
										gamma, n_cycles, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids, n_foods_spawn, model_path, logger,
										debug, cnn_properties)
		write_results_file(data_dir / 'performances' / 'lb_foraging', filename + '_' + model_dirname, results, logger)
	elif test_mode == TestType.CONFIGURATION:
		log_filename = ('test_lb_single_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level_%s-config' % (field_size[0], field_size[1], n_agents, n_foods_spawn,
																										food_level, model_name) + '_' + now.strftime("%Y%m%d-%H%M%S"))
		logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
							level=logging.INFO, encoding='utf-8')
		logger = logging.getLogger('INFO')
		err_logger = logging.getLogger('ERROR')
		handler = logging.StreamHandler(sys.stderr)
		handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
		err_logger.addHandler(handler)
		model_path = (models_dir / ('lb_coop_single%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
					  ('%d-agents' % n_agents))
		filename = 'test_lb_foraging_%d_foods_config' % n_foods_spawn
		food_pos = [(1, 7), (3, 0), (3, 1), (5, 4), (6, 6), (7, 1)]
		obj_loc = (7, 1)
		agents_pos = [(2, 6), (3, 7)]
		results = test_configuration(n_agents, player_level, field_size, n_foods, sight, max_steps, food_level, food_locs, n_layers, layer_sizes,
									 gamma, use_render, dueling_dqn, use_ddqn, use_cnn, agent_ids, n_foods_spawn, model_path, model_dirname, food_pos, obj_loc,
									 agents_pos, logger, cnn_properties)
		write_results_file(data_dir / 'performances' / 'lb_foraging', filename, results, logger)
	else:
		print('[ARGS ERROR] Invalid testing mode, please review test mode.')
		return
	

if __name__ == '__main__':
	main()
