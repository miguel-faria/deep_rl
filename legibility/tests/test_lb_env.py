#! /usr/bin/env python
import time

import numpy as np
import flax.linen as nn
import yaml

from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv
from dl_algos.single_model_madqn import SingleModelMADQN
from itertools import product
from pathlib import Path

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
RNG_SEED = 123456789


def main():
	
	n_agents = 2
	player_level = 1
	field_size = (8, 8)
	n_foods = 8
	n_food_spawn = 2
	sight = field_size[0]
	max_steps = 5000
	food_level = 2
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	
	n_layers = 2
	layer_sizes = [256, 256]
	buffer_size = 10000
	gamma = 0.95
	beta = 2.5
	use_gpu = True
	dueling_dqn = True
	use_ddqn = True
	use_cnn = True
	use_tensorboard = False
	tensorboard_details = []
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1]) + '_food_locs'
		if dict_idx in config_params.keys():
			food_locs = config_params[dict_idx]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]

	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	optim_dir = (models_dir / 'lb_coop_legible_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
				 ('%d-foods_%d-food-level' % (n_food_spawn, food_level)) / 'best')
	loc = food_locs[0]
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs, food_locs[0],
							 render_mode=['rgb_array', 'human'], grid_observation=True, agent_center=True)
	# env = LBForagingEnv(n_agents, player_level, field_size, n_foods, sight, max_steps, True, render_mode=['rgb_array', 'human'], grid_observation=True)
	# legible_dqn = SingleModelMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.observation_space[0],
	# 							   use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard, tensorboard_details)
	# legible_dqn.load_model(('food_%dx%d' % (loc[0], loc[1])), optim_dir, None,
	# 					  env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape), False)
	# print([agent.position for agent in env.players])
	# print([food.position for food in env.foods])
	# print(env.field)
	# print('\n')
	# print([agent.position for agent in env.players])
	# print([food.position for food in env.foods])
	# print(env.field)
	# print('\n')
	# print(state)
	
	agent_actions = np.zeros((2, 100, 1000))
	# for cycle in range(100):
	env.seed(RNG_SEED)
	env.spawn_food(n_food_spawn, food_level)
	env.spawn_players()
	state, *_ = env.reset()
	print(env.field)
	print(env.foods)
	print(state[0].shape)
	agent_state = state[0]
	state_shape = agent_state.shape
	# print(state_shape)
	s_reshape = agent_state.reshape((1, *state_shape[1:], state_shape[0]))
	# print(s_reshape.shape)
	# print(s_reshape[0, 0, 0])
	# env.render()
	time.sleep(0.5)
		# for i in range(1000):
		#
		# 	print('Iteration: %d' % (i + 1))
		# 	actions = env.action_space.sample()
		# 	agent_actions[0, cycle, i] = actions[0]
		# 	agent_actions[1, cycle, i] = actions[1]
		# 	print(' '.join([ACTION_MAP[action] for action in actions]))
		# 	state, rewards, dones, _, info = env.step(actions)
		# 	print(state, rewards)
		# 	env.render()
		# 	input()
	
	# for a in range(2):
	# 	for idx in range(1000):
	# 		print((agent_actions[a, :, idx] == agent_actions[a, 0, idx]).all())

if __name__ == '__main__':
	main()
