#! /usr/bin/env python
import numpy as np
import yaml

from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv
from itertools import product
from pathlib import Path

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
KEY_MAP = {'w': 1, 's': 2, 'a': 3, 'd': 4, 'q': 0, 'e': 5}
RNG_SEED = 123456789


def main():
	
	n_agents = 2
	player_level = 1
	field_size = (8, 8)
	n_foods = 8
	n_foods_spawn = 6
	sight = 7
	max_steps = 5000
	food_level = 2
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = config_params['food_locs'][dict_idx]
			food_confs = config_params['food_confs'][dict_idx][(n_foods_spawn - 1)]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	obj_food = food_locs[2]
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs, food_locs[1],
							 render_mode=['rgb_array', 'human'], use_encoding=False, agent_center=True, grid_observation=True)
	# env = LBForagingEnv(n_agents, player_level, field_size, n_foods, sight, max_steps, True, render_mode=['rgb_array', 'human'], grid_observation=True,
	# 					agent_center=False)
	# n_food_spawn = np.random.choice(range(n_foods))
	
	if food_confs is not None:
		food_conf = food_confs[np.random.choice(range(len(food_confs)))]
		locs = [food for food in food_locs if food != obj_food]
		foods_spawn = [locs[idx] for idx in food_conf]
		env.food_spawn_pos = foods_spawn
	print(obj_food, env.food_spawn_pos)
	env.set_objective(obj_food)
	env.seed(seed=123456789)
	env.spawn_food(n_foods_spawn, food_level)
	env.spawn_players(player_level)
	print('Food objective is (%d,%d)' % (obj_food[0], obj_food[1]))
	state, *_ = env.reset(seed=123456789)
	print(state, state.shape)
	for s in state:
		for layer in s:
			print(layer)
			print('\n')
		print('\n\n')
	print(env.field)
	env.render()
	input()
	
	for i in range(100):

		print('Iteration: %d' % (i + 1))
		actions = []
		for a_idx in range(n_agents):
			valid_action = False
			while not valid_action:
				human_input = input("Action for agent %d:\t" % (a_idx + 1))
				action = int(KEY_MAP[human_input])
				if action < 6:
					valid_action = True
					actions.append(action)
				else:
					print('Action ID must be between 0 and 5, you gave ID %d' % action)

		print(' '.join([ACTION_MAP[action] for action in actions]))
		state, rewards, done, timeout, info = env.step(actions)
		print(state)
		if done or timeout:
			state, *_ = env.reset()
		env.render()
		input()


if __name__ == '__main__':
	main()
