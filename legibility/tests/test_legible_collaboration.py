#! /usr/bin/env python

import numpy as np
import argparse
import gymnasium
import pickle
import yaml
import itertools
import jax
import logging
import os
import multiprocessing as mp
import threading

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Callable
from gymnasium.spaces import MultiBinary
from agents.agent import Agent
from agents.tom_agent import TomAgent
from statistics import stdev
from math import sqrt
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import Action
from dl_algos.dqn import DQNetwork
from flax.linen import relu


RNG_SEED = 20240729


def run_test_iteration(leader_agent: Agent, tom_agent: TomAgent, n_agents: int, player_level: int, field_dims: Tuple[int, int], max_foods: int, player_sight: int,
                       max_steps: int, foods_lvl: int, rng_seed: int, food_locs: List[Tuple], use_render: bool, use_cnn: bool, max_foods_spawn: int) -> Dict:

	env = FoodCOOPLBForaging(n_agents, player_level, field_dims, max_foods, player_sight, max_steps, True, foods_lvl, rng_seed, food_locs, use_render=use_render,
	                         use_encoding=True, agent_center=True, grid_observation=use_cnn)
	it_results = {}
	rng_gen = np.random.default_rng(rng_seed)
	spawned_foods = [food_locs[idx] for idx in rng_gen.choice(max_foods, size=max_foods_spawn, replace=False)]
	foods_left = spawned_foods.copy()
	n_foods_left = max_foods_spawn
	start_obj = foods_left.pop(rng_gen.integers(max_foods))

	# Setup agents for test
	leader_agent.init_interaction([str(food) for food in spawned_foods])
	tom_agent.init_interaction([str(food) for food in spawned_foods])

	# Setup environment for test
	env.food_spawn_pos = spawned_foods
	env.n_food_spawn = max_foods_spawn
	env.set_objective(start_obj)
	env.spawn_players()
	env.spawn_food(max_foods_spawn, foods_lvl)
	obs, *_ = env.reset()

	timeout = False
	while n_foods_left > 0 and not timeout:
		pass

	return it_results


def eval_legibility(n_runs: int, test_mode: int, logger: logging.Logger, opt_models_dir: Path, leg_models_dir: Path, field_dims: Tuple[int, int], n_agents: int,
                    player_level: int, player_sight: int, max_foods: int, max_foods_spawn: int, food_locs: List[Tuple], foods_lvl: int, max_steps: int, gamma: float,
                    num_layers: int, act_function: Callable, layer_sizes: List[int], use_cnn: bool, use_dueling: bool, use_ddqn: bool, cnn_properties: List = None,
                    run_paralell: bool = False, use_render: bool = False):

	env = FoodCOOPLBForaging(n_agents, player_level, field_dims, max_foods, player_sight, max_steps, True, foods_lvl, RNG_SEED, food_locs, use_render=use_render,
	                         use_encoding=True, agent_center=True, grid_observation=use_cnn)
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	optim_models = {}
	leg_models = {}
	opt_model_names = [fname.name for fname in (opt_models_dir / ('%d-foods_%d-food-level' % (max_foods_spawn, foods_lvl)) / 'best').iterdir()]
	leg_model_names = [fname.name for fname in (leg_models_dir / ('%d-foods_%d-food-level' % (max_foods_spawn, foods_lvl)) / 'best').iterdir()]
	try:
		for loc in food_locs:
			# Find the optimal model name for the food location
			model_name = ''
			for name in opt_model_names:
				if name.find("%sx%s" % (loc[0], loc[1])) != -1:
					model_name = name
					break
			assert model_name != ''
			opt_dqn = DQNetwork(len(Action), num_layers, act_function, layer_sizes, gamma, use_dueling, use_ddqn, use_cnn, cnn_properties)
			opt_dqn.load_model(model_name, opt_models_dir / ('%d-foods_%d-food-level' % (max_foods_spawn, foods_lvl)) / 'best', logger, cnn_shape, True)
			optim_models[str(loc)] = opt_dqn

			# Find the legible model name for the food location
			model_name = ''
			for name in leg_model_names:
				if name.find("%sx%s" % (loc[0], loc[1])) != -1:
					model_name = name
					break
			assert model_name != ''
			leg_dqn = DQNetwork(len(Action), num_layers, act_function, layer_sizes, gamma, use_dueling, use_ddqn, use_cnn, cnn_properties)
			leg_dqn.load_model(model_name, leg_models_dir / ('%d-foods_%d-food-level' % (max_foods_spawn, foods_lvl)) / 'best', logger, cnn_shape, True)
			leg_models[str(loc)] = leg_dqn

	except AssertionError as e:
		logger.error(e)
		return []

	if test_mode == 0:
		leader_agent = Agent(optim_models, RNG_SEED)
		follower_agent = TomAgent(optim_models, RNG_SEED, 1)
	elif test_mode == 1:
		leader_agent = Agent(optim_models, RNG_SEED)
		follower_agent = TomAgent(leg_models, RNG_SEED, 1)
	elif test_mode == 2:
		leader_agent = Agent(leg_models, RNG_SEED)
		follower_agent = TomAgent(optim_models, RNG_SEED, 1)
	else:
		leader_agent = Agent(leg_models, RNG_SEED)
		follower_agent = TomAgent(leg_models, RNG_SEED, 1)

	results = {}
	if run_paralell:
		t_pool = mp.Pool(int(0.75 * mp.cpu_count()))
		pool_results = [t_pool.apply_async(run_test_iteration, args=(leader_agent, follower_agent, n_agents, player_level, field_dims, max_foods, player_sight, max_steps,
		                                                             foods_lvl, RNG_SEED + run, food_locs, use_render, use_cnn, max_foods_spawn)) for run in range(n_runs)]
		t_pool.close()
		for idx in range(len(pool_results)):
			results[idx] = list(pool_results[idx].get())
		t_pool.join()
	else:
		for run in range(n_runs):
			results[run] = run_test_iteration(leader_agent, follower_agent, n_agents, player_level, field_dims, max_foods, player_sight,
			                                  max_steps, foods_lvl, RNG_SEED + run, food_locs, use_render, use_cnn, max_foods_spawn)

	return results


def main():

	parser = argparse.ArgumentParser(description='Testing level based foraging with different legible agent configurations.')

	# Test configuration
	parser.add_argument('--mode', dest='mode', type=int, required=True, choices=[0, 1, 2, 3],
						help='Team composition mode:'
							 '\n\t0 - Optimal agent controls interaction with an optimal follower '
							 '\n\t1 - Legible agent controls interaction with a legible follower '
							 '\n\t2 - Legible agent controls interaction with an optimal follower'
							 '\n\t3 - Optimal agent controls interaction with a legible follower')
	parser.add_argument('--runs', dest='nruns', type=int, required=True, help='Number of trial runs to obtain eval')
	parser.add_argument('--render', dest='render', type=bool, action='store_true', help='Activate the render to see the interaction')
	parser.add_argument('--paralell', dest='paralell', type=bool, action='store_true',
						help='Use paralell computing to speed the evaluation process. (Can\'t be used with render or gpu active)')
	parser.add_argument('--use_gpu', dest='gpu', type=bool, action='store_true', help='Use gpu for matrix computations')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
						help='Directory to store trained models and load optimal models, if left blank stored in default location')
	parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
						help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
	parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')

	# Environment configuration
	parser.add_argument('--n-agents', dest='n_agents', type=int, required=True, help='Number of agents in the foraging environment')
	parser.add_argument('--player-level', dest='player_level', type=int, required=True, help='Level of the agents collecting food')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--n-food', dest='n_foods', type=int, required=True, help='Number of food items in the field')
	parser.add_argument('--food-level', dest='food_level', type=int, required=True, help='Level of the food items')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--n-foods-spawn', dest='n_foods_spawn', type=int, required=True, help='Number of foods to be spawned for training.')

	# Agent configuration
	parser.add_argument('--n-leg-agents', dest='n_leg_agents', type=int, default=1, help='Number of legible agents in the environment')
	parser.add_argument('--architecture', dest='architecture', type=str, required=True, help='DQN architecture to use from the architectures yaml')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')

	args = parser.parse_args()

	# Test parameter
	mode = args.mode
	n_runs = args.nruns
	use_render = args.render
	use_paralell = args.paralell
	use_gpu = args.gpu

	# Environment parameters
	n_agents = args.n_agents
	player_level = args.player_level
	field_lengths = args.field_lengths
	n_foods = args.n_foods
	food_level = args.food_level
	steps_episode = args.max_steps
	n_foods_spawn = args.n_foods_spawn

	# Agents parameters
	n_leg_agents = args.n_leg_agents
	architecture = args.architecture
	gamma = args.gamma
	use_cnn = args.use_cnn
	use_dueling_dqn = args.dueling_dqn
	use_vdn = args.use_vdn

	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.fraction
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
	home_dir = Path(__file__).parent.absolute().parent.absolute()
	log_dir = Path(args.logs_dir) if args.logs_dir != '' else home_dir / 'logs'
	data_dir = Path(args.data_dir) if args.data_dir != '' else home_dir / 'data'
	models_dir = Path(args.models_dir) if args.models_dir != '' else home_dir / 'models'
	log_filename = (('test_lb_coop_legibile_%dx%d-field_mode-%d_%d-foods_%d-food-level' % (field_size[0], field_size[1], mode, food_level, n_foods)) +
	                '_' + now.strftime("%Y%m%d-%H%M%S"))
	leg_models_dir = models_dir / ('lb_coop_legible%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents)
	opt_models_dir = models_dir / ('lb_coop_single%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents)

	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as config_file:
		config_params = yaml.safe_load(config_file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		assert dict_idx in list(config_params['food_locs'].keys())
		food_locs = [tuple(x) for x in config_params['food_locs'][dict_idx]]

	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)

	logger = logging.getLogger('lb_coop_legible_mode_%d_start_%d_foods' % (mode, n_foods_spawn))
	logger.setLevel(logging.INFO)
	file_handler = logging.FileHandler(log_dir / (log_filename + '.log'))
	file_handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
	file_handler.setLevel(logging.INFO)
	logger.addHandler(file_handler)

	test_fields = []

	results = {}


	return 0


if __name__ == '__main__':
	main()
