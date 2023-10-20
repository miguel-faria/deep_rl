#! /usr/bin/env python

import sys
import argparse

import flax.core
import numpy as np
import jax
import flax.linen as nn
import os
import math
import random
import time
import optax
import json
import yaml
import logging

from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_envs.astro_waste.astro_waste_env import AstroWasteEnv, AgentType, Actions
from dl_envs.astro_waste.astro_greedy_human_model import GreedyHumanAgent
from pathlib import Path
from flax.training.train_state import TrainState
from typing import List
from datetime import datetime
from itertools import permutations


RNG_SEED = 21062023
ROBOT_NAME = 'astro'


def get_history_entry(obs: AstroWasteEnv.Observation, actions: List[int], n_agents: int) -> List:
	
	entry = []
	for a_idx in range(n_agents):
		agent = obs.players[a_idx]
		action = actions[a_idx]
		entry += [agent.id, agent.position, agent.orientation, agent.held_objects, action]
	
	return entry


def train_astro_model(agents_ids: List[str], waste_env: AstroWasteEnv, astro_model: DQNetwork, human_model: GreedyHumanAgent, waste_order: List,
					  num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
					  eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000,
					  train_freq: int = 10, summary_frequency: int = 1000, greedy_actions: bool = True, debug_mode: bool = False) -> List:
	
	history = []
	random.seed(rng_seed)
	np.random.seed(rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	robot_idx = agents_ids.index(ROBOT_NAME)
	
	obs, *_ = waste_env.reset()
	if astro_model.cnn_layer:
		astro_model.init_network_states(rng_seed, obs[robot_idx].reshape(1, *obs[robot_idx].shape), optim_learn_rate)
	else:
		astro_model.init_network_states(rng_seed, obs[robot_idx], optim_learn_rate)
	
	start_time = time.time()
	epoch = 0
	n_agents = len(agents_ids)
	episode_rewards = 0
	episode_start = epoch
	
	for it in range(num_iterations):
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		episode_history = []
		done = False
		while not done:
			# interact with environment
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			if rng_gen.random() < eps:
				actions = []
				human_idx = 0
				for a_id in agents_ids:
					if a_id != ROBOT_NAME:
						actions += [human_model.act(waste_env.create_observation())]
						human_idx += 1
					else:
						actions += [waste_env.action_space.sample()]
			else:
				actions = []
				human_idx = 0
				for a_id in agents_ids:
					if a_id != ROBOT_NAME:
						actions += [human_model.act(waste_env.create_observation())]
						human_idx += 1
					else:
						if astro_model.cnn_layer:
							q_values = astro_model.q_network.apply(astro_model.online_state.params, obs[robot_idx].reshape((1, *obs[robot_idx].shape)))[0]
						else:
							q_values = astro_model.q_network.apply(astro_model.online_state.params, obs[robot_idx])
						
						if greedy_actions:
							action = q_values.argmax(axis=-1)
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(waste_env.action_space[0].n), p=pol)
						actions += [int(jax.device_get(action))]
						if astro_model.use_summary:
							astro_model.summary_writer.add_scalar("charts/episodic_q_vals", float(q_values[int(action)]), epoch)
			if debug_mode:
				logger.info('Environment current state')
				logger.info(waste_env.get_env_log())
				logger.info(str(human_model))
				logger.info('Player actions: %s' % str([Actions(act).name for act in actions]))
			next_obs, rewards, terminated, timeout, infos = waste_env.step(actions)
			if debug_mode:
				logger.info('Player rewards: %s' % str(rewards))
			episode_history += [get_history_entry(waste_env.create_observation(), actions, len(agents_ids))]
			episode_rewards += rewards[robot_idx]
			if astro_model.use_summary:
				astro_model.summary_writer.add_scalar("charts/reward", rewards[robot_idx], epoch)
				# astro_model.summary_writer.add_text("logs/observation", str(obs), epoch)
				# astro_model.summary_writer.add_text("logs/action", str(actions[robot_idx]), epoch)
				# astro_model.summary_writer.add_text("logs/next_observation", str(next_obs), epoch)
			
			if terminated:
				finished = np.ones(n_agents)
			else:
				finished = np.zeros(n_agents)
			
			# store new samples
			astro_model.replay_buffer.add(obs[robot_idx], next_obs[robot_idx], np.array(actions[robot_idx]), rewards[robot_idx],
										  finished[robot_idx], [infos])
			obs = next_obs
			
			# update Q-network and target network
			if epoch > warmup:
				if epoch % train_freq == 0:
					astro_model.update_online_model(batch_size, epoch, start_time, summary_frequency)
				
				if epoch % target_freq == 0:
					astro_model.update_target_model(tau)
				
			epoch += 1
			sys.stdout.flush()
			if terminated or timeout:
				if astro_model.use_summary:
					astro_model.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it)
					astro_model.summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it)
					astro_model.summary_writer.add_scalar("charts/epsilon", eps, it)
				obs, *_ = waste_env.reset()
				episode_rewards = 0
				episode_start = epoch
				done = True
				history += [episode_history]
				human_model.reset(waste_order, dict([(idx, waste_env.objects[idx].position) for idx in range(waste_env.n_objects)]))
		
	return history


def main():
	parser = argparse.ArgumentParser(description='Train DQN model for Astro waste disposal game.')

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
	parser.add_argument('--game-levels', dest='game_levels', type=str, required=True, nargs='+', help='Level to train Astro in.')
	parser.add_argument('--max-env-steps', dest='max_env_steps', type=int, required=True, help='Maximum number of steps for environment timeout')
	parser.add_argument('--field-size', dest='field_size', type=int, required=True, nargs='+', help='Number of rows and cols in field')
	parser.add_argument('--has-slip', dest='has_slip', action='store_true', help='')
	parser.add_argument('--force-facing', dest='need_facing', action='store_true', help='')
	parser.add_argument('--agent-centered', dest='centered_obs', action='store_true', help='')
	parser.add_argument('--use-encoding', dest='use_encoding', action='store_true', help='')

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
	
	# Astro environment args
	game_levels = args.game_levels
	field_size = tuple(args.field_size) if len(args.field_size) == 2 else tuple([args.field_size[0], args.field_size[0]])
	n_players = args.n_agents
	has_slip = args.has_slip
	max_episode_steps = args.max_env_steps
	facing = args.need_facing
	centered_obs = args.centered_obs
	use_encoding = args.use_encoding

	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	if not use_gpu:
		jax.default_device(jax.devices("cpu")[0])
	
	now = datetime.now()
	home_dir = Path(__file__).parent.absolute().parent.absolute()
	log_dir = home_dir / 'logs'
	models_dir = home_dir / 'models'
	configs_dir = Path(__file__).parent.absolute() / 'dl_envs' / 'astro_waste' / 'data' / 'configs'
	model_path = models_dir / 'astro_disposal_dqn' / now.strftime("%Y%m%d-%H%M%S")
	rng_gen = np.random.default_rng(RNG_SEED)

	for game_level in game_levels:
		log_filename = ('train_astro_disposal_dqn_%s' % game_level)
		logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
							level=logging.INFO)
		logger = logging.getLogger('INFO')
		err_logger = logging.getLogger('ERROR')
		handler = logging.StreamHandler(sys.stderr)
		handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
		err_logger.addHandler(handler)
		Path.mkdir(model_path, parents=True, exist_ok=True)
		with open(configs_dir / 'layouts' / (game_level + '.yaml')) as config_file:
			n_objects = len(yaml.safe_load(config_file)['objects'])
		
		logger.info('#######################################')
		logger.info('Starting Astro Waste Disposal DQN Train')
		logger.info('#######################################')
		logger.info('Level %s setup' % game_level)
		env = AstroWasteEnv(field_size, game_level, n_players, has_slip, n_objects, max_episode_steps, RNG_SEED, facing, use_cnn, centered_obs, use_encoding)
		obs, *_ = env.reset(seed=RNG_SEED)
		human_agents = [env.players[idx] for idx in range(n_agents) if env.players[idx].agent_type == AgentType.HUMAN]
		agents_id = [agent.id for agent in env.players]
		
		logger.info('Getting human behaviour model')
		# if game_level == 'level_one':
		# 	human_filename = 'filtered_human_logs_lvl_1.csv'
		# elif game_level == 'level_two':
		# 	human_filename = 'filtered_human_logs_lvl_2.csv'
		# else:
		# 	human_filename = 'filtered_human_logs_lvl_1.csv'
		# human_action_log = Path(overcooked_file).parent / 'data' / 'study_logfiles' / human_filename
		# human_model = extract_human_model(human_action_log)
		human_agent = GreedyHumanAgent(human_agents[0].position, human_agents[0].orientation, human_agents[0].id,
									   dict([(idx, env.objects[idx].position) for idx in range(n_objects)]), RNG_SEED, env.field)
		
		logger.info('Train setup')
		waste_idx = []
		for obj in env.objects:
			waste_idx.append(env.objects.index(obj))
		waste_seqs = list(permutations(waste_idx))
		waste_order = list(rng_gen.choice(np.array(waste_seqs)))
		human_agent.waste_order = waste_order
		
		logger.info('Creating DQN and starting train')
		tensorboard_details[0] = tensorboard_details[0] + '/astro_disposal_' + game_level + '_' + now.strftime("%Y%m%d-%H%M%S")
		tensorboard_details += ['astro_' + game_level]
		astro_dqn = DQNetwork(env.action_space.n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.observation_space[0], use_gpu, dueling_dqn, use_ddqn,
							  use_cnn, False, use_tensorboard, tensorboard_details)
		history = train_astro_model(agents_id, env, astro_dqn, human_agent, waste_order, n_iterations, max_episode_steps * n_iterations, batch_size, learn_rate,
									target_update_rate, initial_eps, final_eps, eps_type, RNG_SEED, logger, eps_decay, warmup, target_freq, train_freq,
									tensorboard_freq, debug_mode=debug)

		logger.info('Saving model and history list')
		Path.mkdir(model_path, parents=True, exist_ok=True)
		astro_dqn.save_model(game_level, model_path, logger)
		obs_path = model_path / (game_level + '.json')
		with open(obs_path, "w") as of:
			of.write(json.dumps(history))


if __name__ == '__main__':
	main()
