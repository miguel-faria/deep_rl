#! /usr/bin/env python

from dl_envs.lb_foraging_coop import LimitedCOOPLBForaging, FoodCOOPLBForaging
from gymnasium.spaces.multi_discrete import MultiDiscrete

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}


def main():
	
	n_agents = 2
	player_level = 1
	field_size = (10, 10)
	n_foods = 6
	sight = field_size[0]
	max_steps = 5000
	food_level = 2
	
	env = LimitedCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level)
	print(len(env.action_space),  MultiDiscrete(env.observation_space[0].high - env.observation_space[0].low).nvec)
	
	state, _, _, _ = env.reset()
	
	for i in range(1000):
		
		print('Iteration: %d' % (i + 1))
		actions = env.action_space.sample()
		
		print(' '.join([ACTION_MAP[action] for action in actions]))
		state, _, _, _ = env.step(actions)


if __name__ == '__main__':
	main()
