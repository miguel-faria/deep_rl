#! /usr/bin/env python
import numpy as np
import yaml

from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv
from itertools import product
from pathlib import Path

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
RNG_SEED = 123456789


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
	
	# env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs, food_locs[3],
	# 						 render_mode=['rgb_array', 'human'])
	env = LBForagingEnv(n_agents, player_level, field_size, n_foods, sight, max_steps, True, render_mode=['rgb_array', 'human'], grid_observation=True)
	# n_food_spawn = np.random.choice(range(n_foods))
	env.spawn_food(n_foods, food_level)
	env.spawn_players(player_level)
	# print([agent.position for agent in env.players])
	# print([food.position for food in env.foods])
	# print(env.field)
	# print('\n')
	state, *_ = env.reset()
	# print([agent.position for agent in env.players])
	# print([food.position for food in env.foods])
	# print(env.field)
	# print('\n')
	print(state)
	print(env.render())
	input()
	
	for i in range(1000):

		print('Iteration: %d' % (i + 1))
		actions = [np.random.choice(range(6)) for _ in range(n_agents)]

		print(' '.join([ACTION_MAP[action] for action in actions]))
		state, rewards, dones, _, info = env.step(actions)
		print(state, rewards)
		env.render()
		input()


if __name__ == '__main__':
	main()
