#! /usr/bin/env python
import numpy as np
import yaml

from dl_envs.lb_foraging_coop import LimitedCOOPLBForaging, FoodCOOPLBForaging
from gymnasium.spaces.multi_discrete import MultiDiscrete
from itertools import product
from pathlib import Path

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}


def main():
	
	
	n_agents = 2
	player_level = 1
	field_size = (8, 8)
	n_foods = 8
	sight = field_size[0]
	max_steps = 5000
	food_level = 2
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1]) + '_food_locs'
		if dict_idx in config_params.keys():
			food_locs = config_params[dict_idx]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, 21459786, food_locs)
	env.spawn_players(player_level)
	n_food_spawn = np.random.choice(range(n_foods))
	env.spawn_food(n_food_spawn, food_level)
	print(food_locs)
	print(env.food_spawn_pos if n_food_spawn < n_foods else food_locs)
	print(env.field)
	state, _, _, _ = env.reset()
	print(state)
	
	for i in range(100):

		print('Iteration: %d' % (i + 1))
		actions = [np.random.choice(env.action_space[idx].n) for idx in range(n_agents)]

		print(' '.join([ACTION_MAP[action] for action in actions]))
		state, _, _, _ = env.step(actions)
		print(state)


if __name__ == '__main__':
	main()
