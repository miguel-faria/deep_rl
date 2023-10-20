#! /usr/bin/env python

import numpy as np
import yaml

from dl_envs.astro_waste.astro_waste_env import AstroWasteEnv, PlayerState, ObjectState, Actions
from pathlib import Path


RNG_SEED = 18102023
ACTION_MAP = {'w': Actions.UP, 's': Actions.DOWN, 'a': Actions.LEFT, 'd': Actions.RIGHT, 'q': Actions.STAY}


def main():
	
	field_size = (15, 15)
	layout = 'cramped_room'
	n_players = 2
	has_slip = False
	n_objects = 4
	max_episode_steps = 500
	facing = True
	layer_obs = True
	centered_obs = False
	encoding = False
	env = AstroWasteEnv(field_size, layout, n_players, has_slip, n_objects, max_episode_steps, RNG_SEED, facing, layer_obs, centered_obs, encoding)
	state, *_ = env.reset()
	print(env.get_filled_field())
	
	for i in range(10):

		print('Iteration: %d' % (i + 1))
		# actions = [np.random.choice(range(6)) for _ in range(n_players)]
		actions = []
		for idx in range(n_players):
			print('Player %s at (%d, %d)' % (env.players[idx].id, *env.players[idx].position))
			action = input('%s action: ' % env.players[idx].id)
			actions += [int(ACTION_MAP[action])]

		print(' '.join([Actions(action).name for action in actions]))
		state, rewards, dones, _, info = env.step(actions)
		print(env.get_filled_field())
		for layer in state[1, :]:
			print(layer)
			print('\n')
	



if __name__ == '__main__':
	main()
