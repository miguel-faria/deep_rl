#! /usr/bin/env python
import numpy as np

from dl_envs.pursuit.pursuit_env import PursuitEnv, Action, TargetPursuitEnv


RNG_SEED = 12072023


def main():
	
	# hunters = ['hunter_1', 'hunter_2', 'hunter_3', 'hunter_4']
	# preys = ['prey_1', 'prey_2']
	hunter_ids = ['h1', 'h2']
	prey_ids = ['p1', 'p2']
	field_size = (10, 10)
	hunter_sight = 10
	max_steps = 50
	n_catch = 3
	it = 0
	hunters = []
	preys = []
	n_hunters = len(hunter_ids)
	n_preys = len(prey_ids)
	for idx in range(n_hunters):
		hunters += [(hunter_ids[idx], 1)]
	for idx in range(n_preys):
		preys += [(prey_ids[idx], 0)]
	
	# env = PursuitEnv(hunters, preys, field_size, hunter_sight, n_catch, max_steps)
	env = TargetPursuitEnv(hunters, preys, field_size, hunter_sight, prey_ids[0], n_catch, max_steps, use_layer_obs=True)
	env.seed(RNG_SEED)
	n_hunters = len(hunters)
	n_preys = len(preys)
	init_pos_hunter = {}
	for hunter in hunter_ids:
		hunter_idx = hunter_ids.index(hunter)
		init_pos_hunter[hunter] = (hunter_idx // n_hunters, hunter_idx % n_hunters)
	init_pos_prey = {}
	for prey in prey_ids:
		prey_idx = prey_ids.index(prey)
		init_pos_prey[prey] = (max(field_size[0] - (prey_idx // n_preys) - 1, 0), max(field_size[1] - (prey_idx % n_preys) - 1, 0))
	env.spawn_hunters(init_pos_hunter)
	env.spawn_preys(init_pos_prey)
	state, *_ = env.reset()
	env.render()
	input()
	
	for i in range(max_steps * 2):
		
		print('Iteration: %d\tStep: %d' % (it + 1, i + 1))
		actions = []
		for hunter_id in hunters:
			print(env.agents[hunter_id[0]].pos)
			# if env.agents[hunter_id].pos[1] < field_size[1] - 1:
			# 	actions += [Action.RIGHT]
			# elif env.agents[hunter_id].pos[0] < field_size[0] - 1:
			# 	actions += [Action.DOWN]
			# elif env.agents[hunter_id].pos[1] > 0:
			# 	actions += [Action.LEFT]
			# else:
			# 	actions += [Action.UP]
			actions += [np.random.choice(len(Action))]
		for _ in preys:
			actions += [np.random.choice(len(Action))]
		
		print(' '.join([str(Action(action)) for action in actions]))
		state, rewards, finished, timeout, _ = env.step(actions)
		print(state[0].shape)
		print(rewards, finished, timeout)
		print(env.field)

		if finished:
			print('Finished\n\n')
			state, *_ = env.reset()
			it += 1


if __name__ == '__main__':
	main()
