#! /usr/bin/env python

import sys
import argparse
import numpy as np
import jax
import flax.linen as nn
import os
import math
import random
import time
import optax
import json

from overcooked_ai_py import __file__ as overcooked_file
from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_envs.astro_waste_disposal import OvercookedGame
from overcooked_ai_py.mdp.actions import Direction, Action
from get_human_model import extract_human_model
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from flax.training.train_state import TrainState
from typing import Callable, List, Tuple, Set
from termcolor import colored
from datetime import datetime


RNG_SEED = 21062023
STATE_LEN = 8
OBJ_LEN = 4
ACTION_DIM = 6
COEF = 1.5
LVL_TRAIN = 'level_one'


def human_action(obs: List, human_model: Callable, env_objs: List, num_agents: int) -> int:
	
	human_obs = get_human_obs(obs, num_agents)
	human_pos = human_obs[:2]
	human_or = human_obs[2:4]


def get_human_obs(obs: List, num_agents: int) -> List:
	
	human_obs = []
	for a_idx in range(num_agents):
		pos_and_or = obs[STATE_LEN*a_idx:STATE_LEN*a_idx+4]
		obj = obs[STATE_LEN*a_idx+4:STATE_LEN*(a_idx+1)]
		if all([elem == -1 for elem in obj]):
			human_obs += [*pos_and_or, 0]
		else:
			human_obs += [*pos_and_or, 1]
			
	for obj_idx in range(num_agents*STATE_LEN, len(obs), OBJ_LEN):
		pos = obs[obj_idx:obj_idx+2]
		obj_status = obs[obj_idx+3]
		human_obs += [*pos, obj_status]
	
	return human_obs


def get_env_action(action: int) -> Tuple:
	
	if action == 0:
		return Direction.NORTH
	elif action == 1:
		return Direction.SOUTH
	elif action == 2:
		return Direction.WEST
	elif action == 3:
		return Direction.EAST
	elif action == 4:
		return Action.INTERACT
	else:
		return Action.STAY


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(x) for x in obs])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def convert_orientation(orientation: Tuple) -> List[int]:
	
	one_hot_or = [0] * 4
	
	if orientation == Direction.NORTH:
		one_hot_or[0] = 1
	elif orientation == Direction.SOUTH:
		one_hot_or[1] = 1
	elif orientation == Direction.EAST:
		one_hot_or[2] = 1
	else:
		one_hot_or[3] = 1
	
	return one_hot_or


def train_astro_model(agents_ids: List[str], waste_env: OvercookedGame, astro_model: DQNetwork, human_model: List[Callable], num_iterations: int,
					  batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float, eps_type: str, rng_seed: int,
					  exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000, train_freq: int = 10, summary_frequency: int = 1000) -> List:
	def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, iteration: int, max_iterations: int):
		
		if update_type == 1:
			return max(((final_eps - init_eps) / (max_iterations * decay_rate)) * iteration + init_eps, end_eps)
		elif update_type == 2:
			return max(decay_rate ** iteration * init_eps, end_eps)
		elif update_type == 3:
			return max((1 / (1 + decay_rate * iteration)) * init_eps, end_eps)
		elif update_type == 4:
			return max((decay_rate * math.sqrt(iteration)) * init_eps, end_eps)
		else:
			print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
			return max((1 / (1 + decay_rate * iteration)) * init_eps, end_eps)
	
	# recorded_obs = set()
	history = []
	random.seed(rng_seed)
	np.random.seed(rng_seed)
	rng_gen = np.random.default_rng(rng_seed)
	key = jax.random.PRNGKey(rng_seed)
	key, q_key = jax.random.split(key, 2)
	
	obs, _, _, _ = waste_env.reset()
	if astro_model.online_state is None:
		astro_model.online_state = TrainState.create(
			apply_fn=astro_model.q_network.apply,
			params=astro_model.q_network.init(q_key, obs),
			tx=optax.adam(learning_rate=optim_learn_rate),
		)
	if astro_model.target_params is None:
		astro_model.target_params = astro_model.q_network.init(q_key, obs)
	
	astro_model.q_network.apply = jax.jit(astro_model.q_network.apply)
	astro_model.target_params = optax.incremental_update(astro_model.online_state.params, astro_model.target_params, 1.0)
	
	start_time = time.time()
	epoch = 0
	n_agents = len(agents_ids)
	episode_rewards = 0
	episode_start = epoch
	
	for it in range(num_iterations):
		print("Iteration %d out of %d" % (it + 1, num_iterations))
		episode_history = []
		done = False
		while not done:
			print("Epoch %d" % (epoch + 1))
			obs_dqn = waste_env.get_state_dqn()
			
			# interact with environment
			robot_idx = agents_ids.index('robot')
			eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			if rng_gen.random() < eps:
				actions = []
				human_idx = 0
				for a_id in agents_ids:
					if a_id != 'robot':
						actions += [human_model[human_idx]([get_human_obs(obs, len(agents_ids))])[0]]
						human_idx += 1
					else:
						# possible_actions = [Direction.NORTH, Direction.SOUTH, Direction.WEST, Direction.EAST, Action.INTERACT, Action.STAY]
						actions += [rng_gen.choice(ACTION_DIM)]
			else:
				actions = []
				human_idx = 0
				for a_id in agents_ids:
					if a_id != 'robot':
						actions += [human_model[human_idx]([get_human_obs(obs, len(agents_ids))])[0]]
						human_idx += 1
					else:
						q_values = astro_model.q_network.apply(astro_model.online_state.params, obs_dqn)
						action = q_values.argmax(axis=-1)
						actions += [int(jax.device_get(action))]
			actions_env = [get_env_action(act) for act in actions]
			next_obs, rewards, finished, infos = waste_env.step(actions_env)
			next_obs_dqn = waste_env.get_state_dqn()
			episode_history += [get_history_entry(obs, actions, len(agents_ids))]
			episode_rewards += rewards[robot_idx]
			if astro_model.use_summary:
				astro_model.summary_writer.add_scalar("charts/reward", rewards[robot_idx], epoch)
				# astro_model.summary_writer.add_text("logs/observation", str(obs), epoch)
				# astro_model.summary_writer.add_text("logs/action", str(actions[robot_idx]), epoch)
				# astro_model.summary_writer.add_text("logs/next_observation", str(next_obs), epoch)
			
			# store new samples
			real_next_obs = next_obs.copy()
			astro_model.replay_buffer.add(obs_dqn, next_obs_dqn, np.array(actions[robot_idx]), rewards[robot_idx], finished[robot_idx], infos)
			obs = next_obs
			
			# update Q-network and target network
			if epoch > warmup:
				if epoch % train_freq == 0:
					astro_model.update_online_model(batch_size, epoch, start_time, summary_frequency)
				
				if epoch % target_freq == 0:
					astro_model.target_params = optax.incremental_update(astro_model.online_state.params, astro_model.target_params, tau)
				
			epoch += 1
			sys.stdout.flush()
			if all(finished):
				if astro_model.use_summary and all(finished):
					astro_model.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it)
					astro_model.summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it)
					astro_model.summary_writer.add_scalar("charts/epsilon", eps, epoch)
				obs, _, _, _ = waste_env.reset()
				episode_rewards = 0
				episode_start = epoch
				done = True
				history += [episode_history]
		
	return history


def main():
	parser = argparse.ArgumentParser(description='Train DQN model for Astro waste disposal game.')

	# Multi-agent DQN params
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

	# Train parameters
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.95, help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
						help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')

	# Environment parameters
	parser.add_argument('--game-levels', dest='game_levels', type=str, required=True, nargs='+', help='Level to train Astro in.')
	parser.add_argument('--max-env-steps', dest='max_env_steps', type=int, required=True, help='')

	args = parser.parse_args()
	n_layers = args.n_layers
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	n_iterations = args.n_iterations
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	learn_rate = args.learn_rate
	target_learn_rate = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	game_levels = args.game_levels
	max_env_steps = args.max_env_steps

	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	
	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	model_path = models_dir / 'astro_disposal_dqn' / now.strftime("%Y%m%d-%H%M%S")

	for game_level in game_levels:
		log_filename = ('train_astro_disposal_dqn_%s' % game_level)
	
		sys.stdout = open(log_dir / (log_filename + '_' + now.strftime("%Y%m%d-%H%M%S") + '_log.txt'), 'a')
		sys.stderr = open(log_dir / (log_filename + '_' + now.strftime("%Y%m%d-%H%M%S") + '_err.txt'), 'w')
		
		print('#######################################')
		print('Starting Astro Waste Disposal DQN Train')
		print('#######################################')
		print('Level %s setup' % game_level)
		env = OvercookedGame([game_level], userid=2, max_epochs=max_env_steps)
		env.add_player('human', idx=0, is_human=True)
		env.add_player('robot', idx=1, is_human=False)
		obs, _, _, _ = env.reset()
		
		print('Getting human behaviour model')
		if game_level == 'level_one':
			human_filename = 'filtered_human_logs_lvl_1.csv'
		elif game_level == 'level_two':
			human_filename = 'filtered_human_logs_lvl_2.csv'
		else:
			human_filename = 'filtered_human_logs_lvl_1.csv'
		human_action_log = Path(overcooked_file).parent / 'data' / 'study_logfiles' / human_filename
		human_model = extract_human_model(human_action_log)
		
		print('Train setup')
		agents_id = env.players
		num_agents = len(agents_id)
		num_objs = len(env.get_state()['state']['objects'])
		obs_dims = []
		for _ in range(num_agents):
			obs_dims += [16, 16, 2, 2, 2, 2]		# [x, y] + one_hot(orientation)
		for _ in range(num_objs):
			obs_dims += [17, 17, 2, 2, 2]			# [x, y] + one_hot(status)
		obs_space = MultiDiscrete(np.array(obs_dims))
		
		print('Creating DQN and starting train')
		tensorboard_details[0] = tensorboard_details[0] + '/astro_disposal_' + game_level + '_' + now.strftime("%Y%m%d-%H%M%S")
		tensorboard_details += ['astro_' + game_level]
		astro_dqn = DQNetwork(ACTION_DIM, n_layers, nn.relu, layer_sizes, buffer_size, gamma, obs_space, use_gpu, dueling_dqn, use_ddqn,
							  False, use_tensorboard, tensorboard_details)
		history = train_astro_model(agents_id, env, astro_dqn, [human_model.predict], n_iterations, batch_size, learn_rate, target_learn_rate, initial_eps,
									final_eps, eps_type, RNG_SEED, eps_decay, warmup, target_freq, train_freq, tensorboard_freq)
		
		print('Saving model and history list')
		Path.mkdir(model_path, parents=True, exist_ok=True)
		astro_dqn.save_model(game_level, model_path)
		obs_path = model_path / (game_level + '.json')
		with open(obs_path, "w") as of:
			of.write(json.dumps(history))


if __name__ == '__main__':
	main()
