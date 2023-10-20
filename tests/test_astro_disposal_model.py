#! /usr/bin/env python

import argparse
import os
import sys
import random
import jax
import numpy as np
import flax.linen as nn

from overcooked_ai_py import __file__ as overcooked_file
from dl_algos.dqn import DQNetwork
from dl_envs.astro_waste.astro_waste_disposal import OvercookedGame
from overcooked_ai_py.mdp.actions import Direction, Action
from get_human_model import extract_human_model
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from typing import List, Tuple


RNG_SEED = 4072023
STATE_LEN = 8
OBJ_LEN = 4
ACTION_DIM = 6
MAX_EPOCH = 500


def get_human_obs(obs: List, num_agents: int) -> List:
	human_obs = []
	for a_idx in range(num_agents):
		pos_and_or = obs[STATE_LEN * a_idx:STATE_LEN * a_idx + 4]
		obj = obs[STATE_LEN * a_idx + 4:STATE_LEN * (a_idx + 1)]
		if all([elem == -1 for elem in obj]):
			human_obs += [*pos_and_or, 0]
		else:
			human_obs += [*pos_and_or, 1]
	
	for obj_idx in range(num_agents * STATE_LEN, len(obs), OBJ_LEN):
		pos = obs[obj_idx:obj_idx + 2]
		obj_status = obs[obj_idx + (OBJ_LEN - 1)]
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


def main():
	parser = argparse.ArgumentParser(description='Test DQN model for Astro waste disposal game.')
	
	# Multi-agent DQN params
	parser.add_argument('--nlayers', dest='n_layers', type=int, required=True, help='Number of layers for the neural net in the DQN')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	
	# Environment parameters
	parser.add_argument('--game-levels', dest='game_levels', type=str, required=True, nargs='+', help='Level to train Astro in.')
	
	args = parser.parse_args()
	n_layers = args.n_layers
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	game_levels = args.game_levels
	
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	
	for game_level in game_levels:
		log_filename = ('test_astro_disposal_dqn_%s' % game_level)
		model_path = models_dir / 'astro_disposal_dqn'
		
		sys.stdout = open(log_dir / (log_filename + '_log.txt'), 'a')
		sys.stderr = open(log_dir / (log_filename + '_err.txt'), 'w')
		
		print('#######################################')
		print('Starting Astro Waste Disposal DQN Train')
		print('#######################################')
		print('Level %s setup' % game_level)
		env = OvercookedGame([game_level], userid=2)
		env.add_player('human', idx=0, is_human=True)
		env.add_player('robot', idx=1, is_human=False)
		obs, _, _, _ = env.reset()
		agents_id = env.players
		num_agents = len(agents_id)
		num_objs = len(env.get_state()['state']['objects'])
		
		print('Getting human behaviour model')
		if game_level == 'level_one':
			human_filename = 'filtered_human_logs_lvl_1.csv'
		elif game_level == 'level_two':
			human_filename = 'filtered_human_logs_lvl_2.csv'
		else:
			human_filename = 'filtered_human_logs_lvl_1.csv'
		human_action_log = Path(overcooked_file).parent / 'data' / 'study_logfiles' / human_filename
		human_model = extract_human_model(human_action_log)
		
		print('Loading Astro DQN model')
		obs_dims = []
		for _ in range(num_agents):
			obs_dims += [16, 16, 2, 2, 2, 2]  # [x, y] + one_hot(orientation)
		for _ in range(num_objs):
			obs_dims += [17, 17, 2, 2, 2]  # [x, y] + one_hot(status)
		obs_space = MultiDiscrete(np.array(obs_dims))
		astro_dqn = DQNetwork(ACTION_DIM, n_layers, nn.relu, layer_sizes, buffer_size, gamma, obs_space, use_gpu, False, use_tensorboard, tensorboard_details)
		astro_dqn.load_model((game_level + '.model'), model_path)
		
		print('Setting up and running model test')
		history = []
		game_over = False
		random.seed(RNG_SEED)
		np.random.seed(RNG_SEED)
		rng_gen = np.random.default_rng(RNG_SEED)
		epoch = 0
		while not game_over:
			
			obs_dqn = env.get_state_dqn()
			
			actions = []
			for a_id in agents_id:
				if a_id != 'robot':
					actions += [human_model.predict([get_human_obs(obs, len(agents_id))])[0]]
				else:
					q_values = astro_dqn.q_network.apply(astro_dqn.online_state.params, obs_dqn)
					action = q_values.argmax(axis=-1)
					actions += [int(jax.device_get(action))]
			actions_env = [get_env_action(act) for act in actions]
			next_obs, rewards, finished, infos = env.step(actions_env)
			history += [get_history_entry(obs, actions, len(agents_id))]
			obs = next_obs
			print(obs, get_human_obs(obs, len(agents_id)))
			if all(finished) or epoch >= MAX_EPOCH:
				game_over = True
			
			sys.stdout.flush()
			epoch += 1
		
		print('Test history:')
		print(history)
	

if __name__ == '__main__':
	main()
