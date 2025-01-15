#! /usr/bin/env python

import numpy as np
import argparse
import yaml
import jax
import logging
import os
import multiprocessing as mp
import csv

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Callable
from gymnasium.spaces import MultiBinary
from agents.agent import Agent
from agents.tom_agent import TomAgent
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import Action, Direction
from dl_algos.dqn import DQNetwork
from flax.linen import relu

LEADER_ID = 0
TOM_ID = 1
RNG_SEED = 20240729
CONF = 1.0


def write_results_file(data_dir: Path, filename: str, results: Dict, logger: logging.Logger) -> None:
	try:
		with open(data_dir / (filename + '.csv'), 'w') as results_file:
			headers = ['test_nr', 'num_steps', 'pred_steps', 'average_pred_steps', 'num_caught_foods', 'food_steps', 'num_deadlocks']
			writer = csv.DictWriter(results_file, fieldnames=headers, delimiter=',', lineterminator='\n')
			writer.writeheader()
			for key in results.keys():
				row = {'test_nr': key}
				for header, val in zip(headers[1:], list(results[key])):
					row[header] = results[key][val]
				writer.writerow(row)

	except IOError as e:
		logger.error("I/O error: " + str(e))


def append_results_file(data_dir: Path, filename: str, results: Dict, logger: logging.Logger, test_nr: int) -> None:
	try:
		with open(data_dir / (filename + '.csv'), 'a') as results_file:
			headers = ['test_nr', 'num_steps', 'pred_steps', 'average_pred_steps', 'num_caught_foods', 'food_steps', 'num_deadlocks']
			writer = csv.DictWriter(results_file, fieldnames=headers, delimiter=',', lineterminator='\n')
			row = {'test_nr': test_nr}
			for header, key in zip(headers[1:], list(results.keys())):
				row[header] = results[key]
			writer.writerow(row)

	except IOError as e:
		logger.error("I/O error: " + str(e))


def is_deadlock(history: List, new_state: str, last_actions: Tuple) -> bool:
	
	if len(history) < 3:
		return False
	
	deadlock = True
	# if all([act == Action.NONE for act in last_actions]) or all([act == Action.LOAD for act in last_actions]):
	# 	return False
	#
	# else:
	state_repitition = 0
	for state in history:
		if new_state == state:
			state_repitition += 1
	if state_repitition < 3:
		deadlock = False
	
	return deadlock


def coordinate_agents(env: FoodCOOPLBForaging, predict_task: str, actions: Tuple[int, int]) -> Tuple[int, int]:
	
	objective = env.obj_food
	player_pos = [player.position for player in env.players]
	objective_adj = env.get_adj_pos(objective[0], objective[1])
	
	if all([pos in objective_adj for pos in player_pos]):
		if predict_task == str(objective):
			return Action.LOAD, Action.LOAD
		else:
			return actions
	
	else:
		leader_pos = player_pos[LEADER_ID]
		tom_pos = player_pos[TOM_ID]
		lead_direction = Direction[Action(actions[LEADER_ID]).name].value
		tom_direction = Direction[Action(actions[TOM_ID]).name].value
		next_lead_pos = (leader_pos[0] + lead_direction[0], leader_pos[1] + lead_direction[1])
		next_tom_pos = (tom_pos[0] + tom_direction[0], tom_pos[1] + tom_direction[1])
		if next_lead_pos == next_tom_pos or all([act == Action.LOAD for act in actions]):
			return actions[LEADER_ID], Action.NONE.value
		else:
			return actions
	

def load_models(logger: logging.Logger, opt_models_dir: Path, leg_models_dir: Path, n_foods_spawn: int, food_locs: List[Tuple], foods_lvl: int, num_layers: int, act_function: Callable,
                layer_sizes: List[int], gamma: float, use_cnn: bool, use_dueling: bool, use_ddqn: bool, cnn_shape: Tuple, cnn_properties: List = None) -> Tuple[Dict, Dict]:
	optim_models = {}
	leg_models = {}
	opt_model_names = [fname.name for fname in (opt_models_dir / ('%d-foods_%d-food-level' % (n_foods_spawn, foods_lvl)) / 'best').iterdir()]
	leg_model_names = [fname.name for fname in (leg_models_dir / ('%d-foods_%d-food-level' % (n_foods_spawn, foods_lvl)) / 'best').iterdir()]
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
			opt_dqn.load_model(model_name, opt_models_dir / ('%d-foods_%d-food-level' % (n_foods_spawn, foods_lvl)) / 'best', logger, cnn_shape, True)
			optim_models[str(loc)] = opt_dqn
			
			# Find the legible model name for the food location
			model_name = ''
			for name in leg_model_names:
				if name.find("%sx%s" % (loc[0], loc[1])) != -1:
					model_name = name
					break
			assert model_name != ''
			leg_dqn = DQNetwork(len(Action), num_layers, act_function, layer_sizes, gamma, use_dueling, use_ddqn, use_cnn, cnn_properties)
			leg_dqn.load_model(model_name, leg_models_dir / ('%d-foods_%d-food-level' % (n_foods_spawn, foods_lvl)) / 'best', logger, cnn_shape, True)
			leg_models[str(loc)] = leg_dqn
		
		return optim_models, leg_models
	
	except AssertionError as e:
		logger.error(e)
		return {}, {}


def run_test_iteration(start_optim_models: Dict, start_leg_models: Dict, logger: logging.Logger, test_mode: int, run_n: int, n_agents: int, player_level: int, field_dims: Tuple[int, int],
                       max_foods: int, player_sight: int, max_steps: int, foods_lvl: int, rng_seed: int, food_locs: List[Tuple], use_render: bool, use_cnn: bool,
                       max_foods_spawn: int, opt_models_dir: Path, leg_models_dir: Path, gamma: float, num_layers: int, act_function: Callable, layer_sizes: List[int],
                       use_dueling: bool, use_ddqn: bool, cnn_properties: List = None) -> Dict:
	
	# Initialize the agents for the interaction
	if test_mode == 0:
		leader_agent = Agent(LEADER_ID, start_optim_models, rng_seed)
		tom_agent = TomAgent(TOM_ID, start_optim_models, start_optim_models, rng_seed, 1)
	elif test_mode == 1:
		leader_agent = Agent(LEADER_ID, start_optim_models, rng_seed)
		tom_agent = TomAgent(TOM_ID, start_leg_models, start_optim_models, rng_seed, 1)
	elif test_mode == 2:
		leader_agent = Agent(LEADER_ID, start_leg_models, rng_seed)
		tom_agent = TomAgent(TOM_ID, start_optim_models, start_leg_models, rng_seed, 1)
	else:
		leader_agent = Agent(LEADER_ID, start_leg_models, rng_seed)
		tom_agent = TomAgent(TOM_ID, start_leg_models, start_leg_models, rng_seed, 1)
	
	env = FoodCOOPLBForaging(n_agents, player_level, field_dims, max_foods, player_sight, max_steps, True, foods_lvl, rng_seed, food_locs, use_render=use_render,
	                         use_encoding=True, agent_center=True, grid_observation=use_cnn)
	it_results = {}
	rng_gen = np.random.default_rng(rng_seed)
	spawned_foods = [food_locs[idx] for idx in rng_gen.choice(max_foods, size=max_foods_spawn, replace=False)]
	foods_left = spawned_foods.copy()
	n_foods_left = max_foods_spawn
	start_obj = foods_left.pop(rng_gen.integers(max_foods_spawn))
	task = str(start_obj)
	
	# Setup agents for test
	tasks = [str(food) for food in spawned_foods]
	tasks.sort()
	leader_agent.init_interaction(tasks)
	tom_agent.init_interaction(tasks)
	
	# print(leader_agent.tasks, tom_agent.tasks)
	# print(task, tom_agent.predict_task)
	
	# Setup environment for test
	env.food_spawn_pos = spawned_foods
	env.n_food_spawn = max_foods_spawn
	env.set_objective(start_obj)
	env.spawn_players()
	env.spawn_food(max_foods_spawn, foods_lvl)
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	obs, *_ = env.reset()
	
	recent_states = [''.join([''.join(str(x) for x in p.position) for p in env.players]) + ''.join([''.join(str(x) for x in f.position) for f in env.foods])]
	if use_cnn:
		leader_obs = obs[0].reshape((1, *cnn_shape))
		tom_obs = obs[1].reshape((1, *cnn_shape))
	else:
		leader_obs = obs[0]
		tom_obs = obs[1]
	actions = (leader_agent.action(leader_obs, (leader_obs, Action.NONE), CONF, logger, task),
	           tom_agent.action(tom_obs, (leader_obs, Action.NONE), CONF, logger, tom_agent.predict_task))
	
	timeout = False
	n_steps = 0
	n_pred_steps = []
	steps_food = []
	deadlock_states = []
	n_deadlocks = 0
	act_try = 0
	later_error = 0
	later_food_step = 0
	
	if use_render:
		env.render()
	
	logger.info('Started run number %d:' % (run_n + 1))
	logger.info(env.get_full_env_log())
	while n_foods_left > 1 and not timeout:
		logger.info('Run number %d, step %d: remaining %d foods, predicted objective %s and real objective %s from ' % (run_n + 1, n_steps + 1, n_foods_left, tom_agent.predict_task, task) + str(foods_left))
		n_steps += 1
		last_leader_sample = (leader_obs, actions[0])
		if task != tom_agent.predict_task:
			later_error = n_steps
		obs, _, _, timeout, _ = env.step(actions)
		if use_render:
			env.render()
		current_food_count = np.sum([not food.picked for food in env.foods])
		
		if use_cnn:
			leader_obs = obs[0].reshape((1, *cnn_shape))
			tom_obs = obs[1].reshape((1, *cnn_shape))
		else:
			leader_obs = obs[0]
			tom_obs = obs[1]
		
		if timeout:
			n_pred_steps += [later_error - later_food_step]
			steps_food += [n_steps - later_food_step]
			break
		
		elif current_food_count < n_foods_left:
			n_foods_left = current_food_count
			n_pred_steps += [later_error - later_food_step]
			steps_food += [n_steps - later_food_step]
			later_food_step = n_steps
			later_error = n_steps
			
			if current_food_count > 0:
				# Update tasks remaining and samples
				tasks = [str(food) for food in foods_left]
				tasks.sort()
				tom_agent.reset_inference(tasks)
				last_leader_sample = (leader_obs, Action.NONE)
				recent_states = []
				
				# Update decision models
				optim_models, leg_models = load_models(logger, opt_models_dir, leg_models_dir, n_foods_left, food_locs, foods_lvl, num_layers, act_function, layer_sizes,
				                                       gamma, use_cnn, use_dueling, use_ddqn, cnn_shape, cnn_properties)
				if test_mode == 0:
					leader_agent.goal_models = optim_models
					tom_agent.goal_models = optim_models
					tom_agent.sample_models = optim_models
				elif test_mode == 1:
					leader_agent.goal_models = optim_models
					tom_agent.goal_models = leg_models
					tom_agent.sample_models = optim_models
				elif test_mode == 2:
					leader_agent.goal_models = leg_models
					tom_agent.goal_models = optim_models
					tom_agent.sample_models = leg_models
				else:
					leader_agent.goal_models = leg_models
					tom_agent.goal_models = leg_models
					tom_agent.sample_models = leg_models
				
				# Get next objective
				next_obj = foods_left.pop(rng_gen.integers(n_foods_left))
				task = str(next_obj)
				env.set_objective(next_obj)
		
		current_state = ''.join([''.join(str(x) for x in p.position) for p in env.players]) + ''.join([''.join(str(x) for x in f.position) for f in env.foods])
		if is_deadlock(recent_states, current_state, actions):
			n_deadlocks += 1
			if current_state not in deadlock_states:
				deadlock_states.append(current_state)
			act_try += 1
			actions = (leader_agent.sub_acting(leader_obs, logger, act_try - 1, last_leader_sample, CONF, task),
			           tom_agent.sub_acting(tom_obs, logger, act_try, last_leader_sample, CONF))
			# actions = (leader_agent.action(leader_obs, last_leader_sample, CONF, logger, task), tom_agent.sub_acting(tom_obs, logger, act_try, last_leader_sample, CONF))
		else:
			act_try = 0
			actions = (leader_agent.action(leader_obs, last_leader_sample, CONF, logger, task), tom_agent.action(tom_obs, last_leader_sample, CONF, logger))
		
		actions = coordinate_agents(env, tom_agent.predict_task, actions)
		
		recent_states.append(current_state)
		if len(recent_states) > 3:
			recent_states.pop(0)
	
	env.close()
	logger.info('Run Over!!')
	it_results['n_steps'] = n_steps
	it_results['pred_steps'] = n_pred_steps
	it_results['avg_pred_steps'] = np.mean(n_pred_steps) if len(n_pred_steps) > 0 else 0
	it_results['caught_foods'] = max_foods_spawn - n_foods_left
	it_results['steps_food'] = steps_food
	it_results['deadlocks'] = n_deadlocks
	
	return it_results


def eval_legibility(n_runs: int, test_mode: int, logger: logging.Logger, opt_models_dir: Path, leg_models_dir: Path, field_dims: Tuple[int, int], n_agents: int,
                    player_level: int, player_sight: int, max_foods: int, max_foods_spawn: int, food_locs: List[Tuple], foods_lvl: int, max_steps: int, gamma: float,
                    num_layers: int, act_function: Callable, layer_sizes: List[int], use_cnn: bool, use_dueling: bool, use_ddqn: bool, data_dir: Path,
                    cnn_properties: List = None, run_paralell: bool = False, use_render: bool = False, start_run: int = 0):
	
	env = FoodCOOPLBForaging(n_agents, player_level, field_dims, max_foods, player_sight, max_steps, True, foods_lvl, RNG_SEED, food_locs, use_render=use_render,
	                         use_encoding=True, agent_center=True, grid_observation=use_cnn)
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	
	optim_models, leg_models = load_models(logger, opt_models_dir, leg_models_dir, max_foods_spawn, food_locs, foods_lvl, num_layers, act_function, layer_sizes, gamma, use_cnn,
	                                       use_dueling, use_ddqn, cnn_shape, cnn_properties)
	
	if run_paralell:
		results = {}
		t_pool = mp.Pool(int(0.75 * mp.cpu_count()))
		pool_results = [t_pool.apply_async(run_test_iteration, args=(optim_models, leg_models, logger, test_mode, run, n_agents, player_level, field_dims, max_foods, player_sight,
		                                                             max_steps, foods_lvl, RNG_SEED + run, food_locs, use_render, use_cnn, max_foods_spawn, opt_models_dir,
		                                                             leg_models_dir, gamma, num_layers, act_function, layer_sizes, use_dueling, use_ddqn, cnn_properties)) for run in range(start_run, n_runs)]
		t_pool.close()
		for idx in range(len(pool_results)):
			results[idx] = list(pool_results[idx].get())
		t_pool.join()

		if start_run < 1:
			write_results_file(data_dir / 'performances' / 'lb_foraging',
			                   'test_mode-%d_field_%d-%d_foods-%d_agents-%d' % (test_mode, field_dims[0], field_dims[1], max_foods_spawn, n_agents), results, logger)

		else:
			append_results_file(data_dir / 'performances' / 'lb_foraging',
			                   'test_mode-%d_field_%d-%d_foods-%d_agents-%d' % (test_mode, field_dims[0], field_dims[1], max_foods_spawn, n_agents), results, logger)

	else:
		for run in range(start_run, n_runs):
			results = run_test_iteration(optim_models, leg_models, logger, test_mode, run, n_agents, player_level, field_dims, max_foods, player_sight, max_steps,
			                             foods_lvl, RNG_SEED + run, food_locs, use_render, use_cnn, max_foods_spawn, opt_models_dir, leg_models_dir, gamma, num_layers,
			                             act_function, layer_sizes, use_dueling, use_ddqn, cnn_properties)

			if run < 1:
				write_results_file(data_dir / 'performances' / 'lb_foraging',
				                   'test_mode-%d_field_%d-%d_foods-%d_agents-%d' % (test_mode, field_dims[0], field_dims[1], max_foods_spawn, n_agents), {run: results}, logger)

			else:
				append_results_file(data_dir / 'performances' / 'lb_foraging',
				                    'test_mode-%d_field_%d-%d_foods-%d_agents-%d' % (test_mode, field_dims[0], field_dims[1], max_foods_spawn, n_agents), results, logger, run)

			logger.info('Run %d results: ' % run + str(results))


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
	parser.add_argument('--render', dest='render', action='store_true', help='Activate the render to see the interaction')
	parser.add_argument('--paralell', dest='paralell', action='store_true',
	                    help='Use paralell computing to speed the evaluation process. (Can\'t be used with render or gpu active)')
	parser.add_argument('--use_gpu', dest='gpu', action='store_true', help='Use gpu for matrix computations')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
	                    help='Directory to store trained models and load optimal models, if left blank stored in default location')
	parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
	                    help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
	parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--start-run', dest='start_run', type=int, default=0, help='Starting test run number')
	
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
	start_run = args.start_run
	
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
	use_ddqn = args.use_ddqn
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
	
	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)
	
	logger = logging.getLogger('lb_coop_legible_mode_%d_start_%d_foods' % (mode, n_foods_spawn))
	logger.setLevel(logging.INFO)
	file_handler = logging.FileHandler(log_dir / (log_filename + '.log'))
	file_handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
	file_handler.setLevel(logging.INFO)
	logger.addHandler(file_handler)
	
	eval_legibility(n_runs, mode, logger, opt_models_dir, leg_models_dir, field_size, n_agents, player_level, sight, n_foods, n_foods_spawn, food_locs, food_level,
	                steps_episode, gamma, n_layers, relu, layer_sizes, use_cnn, use_dueling_dqn, use_ddqn, data_dir, cnn_properties, use_paralell, use_render, start_run)
	
	return 0


if __name__ == '__main__':
	main()
