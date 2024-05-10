#! /usr/bin/env python

import argparse
import os
import sys
import jax
import numpy as np
import flax.linen as nn
import yaml

from dl_algos.single_model_madqn import CentralizedMADQN
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from itertools import product
from typing import List


RNG_SEED = 4072023
STATE_LEN = 8
ACTION_DIM = 6
MAX_EPOCH = 500


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
	parser = argparse.ArgumentParser(description='Test DQN model for Astro waste disposal game.')
	
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
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	
	# Testing params
	parser.add_argument('--model-info', dest='model_info', type=str, nargs='+', help='List  with the info required to load the model to test: '
																					 '<model_dirname: str> <model_filename: str>')
	
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
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	agent_ids = args.agent_ids
	tensorboard_freq = args.tensorboard_freq
	player_level = args.player_level
	field_lengths = args.field_lengths
	n_foods = args.n_foods
	food_level = args.food_level
	max_steps = args.max_steps
	use_render = args.use_render
	model_info = args.model_info
	n_foods_spawn = args.n_foods_spawn
	
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	
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
	log_filename = ('test_lb_centralized_madqn_%dx%d-field_%d-agents_%d-foods_%d-food-level_%s' % (field_size[0], field_size[1], n_agents, n_foods_spawn,
																								   food_level, model_name))
	model_path = (models_dir / 'lb_coop_central_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
				  ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / model_dirname)
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1]) + '_food_locs'
		if dict_idx in config_params.keys():
			food_locs = config_params[dict_idx]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	sys.stdout = open(log_dir / (log_filename + '_log.txt'), 'a')
	sys.stderr = open(log_dir / (log_filename + '_err.txt'), 'w')
	
	print('#############################')
	print('Starting LB Foraging DQN Test')
	print('#############################')
	print('Environment setup')
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs)
	
	# DQN model loading
	obs_dims = [field_size[0], field_size[1], *([2] * (food_level + 1))] * n_foods + [field_size[0], field_size[1], *([2] * (player_level + 1))] * n_agents
	obs_dims = obs_dims * n_agents
	food_locs = [(5, 4)]
	for loc in food_locs:
		model_loc = loc
		env.seed(RNG_SEED)
		np.random.seed(RNG_SEED)
		print('Testing for location: %dx%d' % (loc[0], loc[1]))
		print('Setup multi-agent DQN')
		# obs_dims = env.observation_space[0].high - env.observation_space[0].low
		central_madqn = CentralizedMADQN(n_agents, env.action_space[0].n, n_layers, convert_joint_act, nn.relu, layer_sizes, buffer_size, gamma,
										 MultiDiscrete(obs_dims), use_gpu, dueling_dqn, use_ddqn, False, use_tensorboard, tensorboard_details)
		central_madqn.load_model(('food_%dx%d' % (loc[0], loc[1])), model_path)
		
		# Testing cycle
		print('Starting testing')
		history = []
		game_over = False
		# if cycle == 0:
		# 	foods_spawn = n_foods
		# else:
		# 	foods_spawn = rng_gen.choice(range(1, n_foods))
		players_pos = [(1, 0), (7, 6)]
		env.obj_food = loc
		env.spawn_food(n_foods_spawn, food_level)
		env.spawn_players(player_level, players_pos)
		print('Cycle params:')
		print('Agents positions:\t', [p.position for p in env.players])
		print('Number of food spawn:\t%d' % n_foods_spawn)
		print('Objective food:\t(%d, %d)' % (loc[0], loc[1]))
		print('Testing field:')
		print(env.field)
		obs, _, _, _ = env.reset()
		if use_render:
			env.render()
			input()
		
		epoch = 0
		while not game_over:
			joint_obs = np.array(obs).ravel()
			q_values = central_madqn.madqn.q_network.apply(central_madqn.madqn.online_state.params, joint_obs)
			joint_action = q_values.argmax(axis=-1)
			joint_action = jax.device_get(joint_action)
			actions = convert_joint_act(joint_action, n_agents)
			next_obs, rewards, finished, infos = env.step(actions)
			if use_render:
				env.render()
			history += [get_history_entry(obs, actions, len(agent_ids))]
			obs = next_obs
	
			if all(finished) or epoch >= MAX_EPOCH:
				game_over = True
	
			sys.stdout.flush()
			epoch += 1
	
		print('Epochs needed to finish: %d' % epoch)
		print('Test history:')
		print(history)
		env.close()
		print('##########################################')
	

if __name__ == '__main__':
	main()
