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
import time
import gc
import threading

from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_algos.single_model_madqn import LegibleSingleMADQN
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import Action
from pathlib import Path
from itertools import product
from typing import List, Tuple
from datetime import datetime
from gymnasium.spaces import MultiBinary, MultiDiscrete


RNG_SEED = 13042023
TEST_RNG_SEED = 4072023
N_TESTS = 100


def input_callback(env: FoodCOOPLBForaging, stop_flag: threading.Event):
	try:
		while not stop_flag.is_set():
			command = input('Interactive commands:\n\trender - display renderization of the interaction\n\tstop_render - stops the renderization\nCommand: ')
			if command == 'render':
				env.use_render = True
			elif command == 'stop_render':
				if env.use_render:
					env.use_render = False
	
	except KeyboardInterrupt as ki:
		return


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


def get_live_obs_goals(env: FoodCOOPLBForaging) -> Tuple[List, List]:
	
	live_goals = []
	goals_obs = []
	for food in env.foods:
		live_goals.append(str(food.position))
		goals_obs.append(env.make_target_grid_observations(food.position))
		
	return live_goals, goals_obs


def train_legible_dqn(env: FoodCOOPLBForaging, dqn_model: LegibleSingleMADQN, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float,
					  tau: float, initial_eps: float, final_eps: float, eps_type: str, reward_type: str, rng_seed: int, logger: logging.Logger,
					  exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1,
					  cycle: int = 0, greedy_action: bool = True, sofmax_temp: float = 1.0):
		
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)
		stop_thread = threading.Event()
		command_thread = threading.Thread(target=input_callback, args=(env, stop_thread))
		command_thread.start()
		
		# Setup DQNs for training
		obs, _ = env.reset()
		if not dqn_model.agent_dqn.dqn_initialized:
			if dqn_model.agent_dqn.cnn_layer:
				dqn_model.agent_dqn.init_network_states(rng_seed, obs[0].reshape((1, *obs[0].shape)), optim_learn_rate)
			else:
				dqn_model.agent_dqn.init_network_states(rng_seed, obs[0], optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		
		for it in range(num_iterations):
			if env.use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			episode_start = epoch
			episode_history = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				
				# decay exploration
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				# interact with environment
				if rng_gen.random() < eps:
					random_actions = env.action_space.sample()
					actions = random_actions[:dqn_model.n_leg_agents].tolist()
					for a_idx in range(dqn_model.n_leg_agents, env.n_players):
						if dqn_model.agent_dqn.cnn_layer:
							q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_model.goal].params,
																		   obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
						else:
							q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_model.goal].params, obs[a_idx])
						
						if greedy_action:
							action = q_values.argmax(axis=-1)
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						actions.append(action)
				else:
					actions = []
					for a_idx in range(env.n_players):
						if a_idx < dqn_model.n_leg_agents:
							online_params = dqn_model.agent_dqn.online_state.params
						else:
							online_params = dqn_model.optimal_models[dqn_model.goal].params
						
						if dqn_model.agent_dqn.cnn_layer:
							q_values = dqn_model.agent_dqn.q_network.apply(online_params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
						else:
							q_values = dqn_model.agent_dqn.q_network.apply(online_params, obs[a_idx])
						
						if greedy_action:
							action = q_values.argmax(axis=-1)
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						action = jax.device_get(action)
						episode_q_vals += (float(q_values[int(action)]) / dqn_model.num_agents)
						actions += [action]
				logger.info(env.get_env_log() + 'Actions: ' + str([Action(act).name for act in actions]) + '\n')
				actions = np.array(actions)
				
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				episode_history += [dqn_model.get_history_entry(obs, actions)]
				if env.use_render:
					env.render()
				
				# Obtain the legible rewards
				legible_rewards = np.zeros(dqn_model.n_leg_agents)
				live_goals, _ = get_live_obs_goals(env)
				n_goals = env.food_spawn
				if n_goals > 1:
					for a_idx in range(dqn_model.n_leg_agents):
						act_q_vals = np.zeros(n_goals)
						action = actions[a_idx]
						goal_action_q = 0.0
						for g_idx in range(n_goals):
							if dqn_model.agent_dqn.cnn_layer:
								obs_reshape = obs[a_idx].reshape((1, *obs[a_idx].shape))
								q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[live_goals[g_idx]].params, obs_reshape)[0]
							else:
								q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[live_goals[g_idx]].params, obs[a_idx])
							if dqn_model.goal == live_goals[g_idx]:
								goal_action_q = q_vals[action]
							act_q_vals[g_idx] = np.exp(dqn_model.beta * (q_vals[action] - q_vals.mean()) / sofmax_temp)
							# act_q_vals[g_idx] = np.exp(dqn_model.beta * q_vals[action])
						if reward_type == 'reward':
							legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) * max(rewards[a_idx], 1e-3)
						elif reward_type == 'q_vals':
							legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) * goal_action_q
						elif reward_type == 'info':
							legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) + rewards[a_idx]
						else:
							legible_rewards[a_idx] = act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()
						episode_rewards += (legible_rewards[a_idx] / dqn_model.n_leg_agents)
				
				else:
					for a_idx in range(dqn_model.n_leg_agents):
						legible_rewards[a_idx] = rewards[a_idx]
				
				if terminated:
					finished = np.ones(dqn_model.num_agents)
					legible_rewards = legible_rewards / (1 - dqn_model.agent_dqn.gamma)
				else:
					finished = np.zeros(dqn_model.num_agents)

				if dqn_model.agent_dqn.use_summary:
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/legible_reward", sum(legible_rewards) / dqn_model.n_leg_agents,
																  epoch + start_record_epoch)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/reward", sum(rewards[:dqn_model.n_leg_agents]) / dqn_model.n_leg_agents,
																  epoch + start_record_epoch)
				# store new samples
				if dqn_model.use_vdn:
					dqn_model.replay_buffer.add(obs, next_obs, actions, np.hstack((legible_rewards[:dqn_model.n_leg_agents], rewards[dqn_model.n_leg_agents:])),
												finished[0], [infos])
				else:
					for a_idx in range(dqn_model.n_leg_agents):
						dqn_model.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], legible_rewards[a_idx], finished[a_idx], [infos])
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
						episode_len = epoch - episode_start
						dqn_model.agent_dqn.summary_writer.add_scalar("charts/episode_q_vals", episode_q_vals, it + start_record_it)
						dqn_model.agent_dqn.summary_writer.add_scalar("charts/mean_episode_q_vals", episode_q_vals / episode_len, it + start_record_it)
						dqn_model.agent_dqn.summary_writer.add_scalar("charts/episode_return", episode_rewards, it + start_record_it)
						dqn_model.agent_dqn.summary_writer.add_scalar("charts/mean_episode_return", episode_rewards / episode_len, it + start_record_it)
						dqn_model.agent_dqn.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
						dqn_model.agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					logger.debug("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (epoch - episode_start, eps, episode_rewards))
					obs, *_ = env.reset()
					if env.use_render:
						env.render()
					else:
						env.close_render()
					history += [episode_history]
					done = True
					episode_rewards = 0
					episode_q_vals = 0
					episode_start = epoch
					episode_history = []
		
		stop_thread.set()
		return history


# noinspection DuplicatedCode
def main():
	parser = argparse.ArgumentParser(description='Train DQN for LB Foraging with fixed foods in environment')
	
	# Multi-agent DQN params
	parser.add_argument('--n-agents', dest='n_agents', type=int, required=True, help='Number of agents in the environment')
	parser.add_argument('--n-leg-agents', dest='n_leg_agents', type=int, default=1, help='Number of legible agents in the environment')
	parser.add_argument('--n-layers', dest='n_layers', type=int, required=True, help='Number of layers for the neural net in the DQN')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--beta', dest='beta', type=float, required=False, default=0.9, help='Constant that defines how close follow optimal')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+',
						help='Size of each layer of the DQN\'s neural net')
	parser.add_argument('--reward', dest='leg_reward', type=str, required=True, choices=['simple', 'reward', 'q_vals', 'info'],
						help='Type of legible reward signal to use from amongst:\n\t\'simple\' - use just the objective saliency as reward'
							 '\n\t\'reward\' - weight environment reward with objective saliency\n\t\'q_vals\' - weight agent q_vals with objective saliency'
							 '\n\t\'info\' - information based legibility by summing environment reward with objective saliency')
	
	# Train parameters
	parser.add_argument('--max-cycles', dest='max_cycles', type=int, required=True, help='Max number of training cycles.')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.5, help='Decay rate for the exploration update')
	parser.add_argument('--legibility-temp', dest='temp', type=float, required=False, default=0.5,
						help='Temperature parameter for legibility softmax')
	parser.add_argument('--cycle-eps-decay', dest='cycle_eps_decay', type=float, required=False, default=0.95,
						help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
						help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000,
						help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	parser.add_argument('--restart', dest='restart_train', action='store_true',
						help='Flag that signals that train is suppose to restart from a previously saved point.')
	parser.add_argument('--restart-info', dest='restart_info', type=str, nargs='+', required=False, default=None,
						help='List with the info required to recover previously saved model and restart from same point: '
							 '<model_dirname: str> <model_filename: str> <last_cycle: int> Use only in combination with --restart option')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	parser.add_argument('--use-opt-vdn', dest='opt_vdn', action='store_true', help='Signal the use of optimal models with a VDN architecture')
	
	# Environment parameters
	parser.add_argument('--n-players', dest='n_players', type=int, required=True, help='Number of players in the foraging environment')
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
	n_leg_agents = args.n_leg_agents
	n_layers = args.n_layers
	buffer_size = args.buffer_size
	gamma = args.gamma
	beta = args.beta
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_cnn = args.use_cnn
	use_vdn = args.use_vdn
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	leg_reward = args.leg_reward
	
	# Train args
	max_cycles = args.max_cycles
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
	temp = args.temp
	optim_vdn = args.opt_vdn
	
	# LB-Foraging environment args
	n_players = args.n_players
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
	log_filename = (('train_lb_coop_legible%s_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level' % (('_vdn' if use_vdn else ''), field_size[0], field_size[1],
																								   n_players, n_foods_spawn, food_level)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / ('lb_coop_legible%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
				  ('%d-agents' % n_players) / ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / now.strftime("%Y%m%d-%H%M%S"))
	optim_dir = (models_dir / ('lb_coop_single%s_dqn' % ('_vdn' if optim_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
				 ('%d-agents' % n_players) /  ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / 'best')
	optim_models = [fname.name for fname in optim_dir.iterdir()]
	
	with open(data_dir / 'performances' / 'lb_foraging' / ('train_legible%s_performances.yaml' % ('_vdn' if use_vdn else '')),
			  mode='r+', encoding='utf-8') as train_file:
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
	n_cycles = min(number_food_combinations(n_foods - 1, n_foods_spawn - 1) * 2, max_cycles)
	logger.info('Number of cycles: %d' % n_cycles)
	
	####################
	## Training Model ##
	####################
	try:
		logger.info('Starting training for different food locations')
		goals = [str(loc) for loc in food_locs]
		for loc in food_locs:
			logger.info('Training for location: %dx%d' % (loc[0], loc[1]))
			logger.info('Environment setup')
			env = FoodCOOPLBForaging(n_players, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
									 use_encoding=True, agent_center=False, grid_observation=use_cnn, use_render=use_render)
			env.seed(RNG_SEED)
			env.set_objective(loc)
			logger.info('Setup multi-agent DQN')
			agent_action_space = env.action_space[0]
			n_actions = agent_action_space.n
			if use_vdn:
				action_space = MultiDiscrete([agent_action_space.n] * env.n_players)
				agent_madqn = LegibleSingleMADQN(n_agents, n_actions, n_layers, nn.relu, layer_sizes, buffer_size, gamma, beta, action_space, env.observation_space,
												 use_gpu, False, optim_dir, optim_models, goals, str(loc), dueling_dqn, use_ddqn, use_vdn,
												 use_cnn, use_tensorboard, tensorboard_details + ['%df-%dx%d-legible' % (n_foods_spawn, loc[0], loc[1])],
												 n_legible_agents=min(n_leg_agents, n_agents))
			else:
				if isinstance(env.observation_space, MultiBinary):
					obs_space = MultiBinary([*env.observation_space.shape[1:]])
				else:
					obs_space = env.observation_space[0]
				agent_madqn = LegibleSingleMADQN(n_agents, n_actions, n_layers, nn.relu, layer_sizes, buffer_size, gamma, beta, agent_action_space,
												 obs_space, use_gpu, False, optim_dir, optim_models, goals, str(loc), dueling_dqn, use_ddqn,
												 use_vdn, use_cnn, use_tensorboard, tensorboard_details + ['%df-%dx%d-legible' % (n_foods_spawn, loc[0], loc[1])],
												 n_legible_agents=min(n_leg_agents, n_agents))
			
			if restart_train:
				start_cycle = int(restart_info[2])
				logger.info('Load trained model')
				agent_madqn.load_model(restart_info[1], model_path.parent.absolute() / restart_info[0], logger,
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
				env.spawn_players([player_level] * n_agents)
				env.spawn_food(n_foods_spawn, food_level)
				agent_madqn.replay_buffer.reset()
				logger.info('Cycle params:')
				logger.info('Number of food spawn:\t%d' % n_foods_spawn)
				logger.info('Food locations: ' + ', '.join(['(%d, %d)' % pos for pos in ([loc] + env.food_spawn_pos if n_foods_spawn < n_foods else food_locs)]))
				logger.info('Food objective: (%d, %d)' % env.obj_food)
			
				logger.info('Starting train')
				logger.info('Cycle starting epsilon: %f' % cycle_init_eps)
				history = train_legible_dqn(env, agent_madqn, n_iterations, max_steps * n_iterations, batch_size, learn_rate, target_update_rate, cycle_init_eps,
											final_eps, eps_type, leg_reward, RNG_SEED, logger, eps_decay, warmup, train_freq, target_freq, tensorboard_freq,
											cycle, greedy_action=False, sofmax_temp=temp)
				
				# Reset params that determine how foods are spawn
				env.food_spawn_pos = None
				env.food_spawn = 0
				
				if debug:
					logger.info('Saving cycle iteration history')
					if not use_cnn:
						json_path = model_path / ('food_%dx%d_history_centralized.json' % (loc[0], loc[1]))
						with open(json_path, 'a') as json_file:
							json_file.write(json.dumps({('cycle_%d' % (cycle + 1)): history}))
				
					logger.info('Saving model after cycle %d' % (cycle + 1))
					Path.mkdir(model_path, parents=True, exist_ok=True)
					agent_madqn.save_model(('food_%dx%d_cycle_%d' % (loc[0], loc[1], cycle + 1)), model_path, logger)
					
			env.close()
			logger.info('Saving final model')
			agent_madqn.save_model(('food_%dx%d' % (loc[0], loc[1])), model_path, logger)
			
			####################
			## Testing Model ##
			####################
			logger.info('Testing for location: %dx%d' % (loc[0], loc[1]))
			env = FoodCOOPLBForaging(n_players, player_level, field_size, n_foods, sight, max_steps, True, food_level, TEST_RNG_SEED, food_locs,
									 use_encoding=True, agent_center=False, grid_observation=use_cnn)
			failed_history = []
			tests_passed = 0
			env.seed(TEST_RNG_SEED)
			np.random.seed(TEST_RNG_SEED)
			rng_gen = np.random.default_rng(TEST_RNG_SEED)
			for i in range(N_TESTS):
				env.set_objective(loc)
				env.spawn_players([player_level] * n_players)
				env.spawn_food(n_foods_spawn, food_level)
				logger.info('Test number %d' % (i + 1))
				logger.info('Number of food spawn:\t%d' % n_foods_spawn)
				logger.info('Food locations: ' + ', '.join(['(%d, %d)' % pos for pos in ([loc] + env.food_spawn_pos if n_foods_spawn < n_foods else food_locs)]))
				logger.info('Agent positions: ' + ', '.join(['(%d, %d)' % p.position for p in env.players]))
				obs, *_ = env.reset()
				epoch = 0
				agent_reward = [0] * n_players
				test_history = []
				game_over = False
				finished = False
				timeout = False
				while not game_over:
					
					actions = []
					for a_idx in range(env.n_players):
						dqn = agent_madqn.agent_dqn
						if a_idx < agent_madqn.n_leg_agents:
							online_params = agent_madqn.agent_dqn.online_state.params
						else:
							online_params = agent_madqn.optimal_models[agent_madqn.goal].params
							
						if use_cnn:
							q_values = dqn.q_network.apply(online_params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
						else:
							q_values = dqn.q_network.apply(online_params, obs[a_idx])
						pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
						pol = pol / pol.sum()
						action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						action = jax.device_get(action)
						actions += [action]
					actions = np.array(actions)
					logger.info(env.get_env_log() + 'Actions: ' + str([Action(act).name for act in actions]) + '\n')
					next_obs, rewards, finished, timeout, infos = env.step(actions)
					agent_reward = [agent_reward[idx] + rewards[idx] for idx in range(n_agents)]
					test_history += [get_history_entry(env.make_obs_array(), actions, agent_madqn.num_agents)]
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
			
			env.close()
			logger.info('Passed %d tests out of %d for location %dx%d' % (tests_passed, N_TESTS, loc[0], loc[1]))
			logger.info('Failed tests history:')
			logger.info(failed_history)
			
			if (tests_passed / N_TESTS) > train_acc['%s, %s' % (loc[0], loc[1])]:
				logger.info('Updating best model for current loc')
				Path.mkdir(model_path.parent.absolute() / 'best', parents=True, exist_ok=True)
				agent_madqn.save_model(('food_%dx%d' % (loc[0], loc[1])), model_path.parent.absolute() / 'best', logger)
				train_acc['%s, %s' % (loc[0], loc[1])] = tests_passed / N_TESTS
	
			gc.collect()
			
		logger.info('Updating best training performances record')
		with open(data_dir / 'performances' / 'lb_foraging' / ('train_legible%s_performances.yaml' % ('_vdn' if use_vdn else '')),
				  mode='r+', encoding='utf-8') as train_file:
			performance_data = yaml.safe_load(train_file)
			field_idx = str(field_size[0]) + 'x' + str(field_size[1])
			food_idx = str(n_foods_spawn) + '-food'
			performance_data[field_idx][food_idx] = train_acc
			train_file.seek(0)
			yaml.safe_dump(performance_data, train_file)
		
	except KeyboardInterrupt as ks:
		logger.info('Caught keyboard interrupt, cleaning up and closing.')
		with open(data_dir / 'performances' / 'lb_foraging' / ('train_legible%s_performances.yaml' % ('_vdn' if use_vdn else '')),
				  mode='r+', encoding='utf-8') as train_file:
			performance_data = yaml.safe_load(train_file)
			field_idx = str(field_size[0]) + 'x' + str(field_size[1])
			food_idx = str(n_foods) + '-food'
			performance_data[field_idx][food_idx] = train_acc
			train_file.seek(0)
			yaml.safe_dump(performance_data, train_file)
	

if __name__ == '__main__':
	main()
	