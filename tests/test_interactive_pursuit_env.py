#! /usr/bin/env python
import numpy as np

from dl_envs.pursuit.pursuit_env import PursuitEnv, TeamPursuitEnv, Action


RNG_SEED = 12072023
ACTION_MAP = {'w': Action.UP, 's': Action.DOWN, 'a': Action.LEFT, 'd': Action.RIGHT, 'q': Action.STAY}


def main():
	
	# hunters = ['hunter_1', 'hunter_2', 'hunter_3', 'hunter_4']
	# preys = ['prey_1', 'prey_2']
	hunters = [('hunter_1', 0), ('hunter_2', 1)]
	# preys = [('prey_1', 0), ('prey_2', 0), ('prey_3', 0)]
	preys = [('prey_1', 0), ('prey_2', 0)]
	target = 'prey_1'
	field_size = (10, 10)
	hunter_sight = 10
	max_steps = 50
	n_catch = 2
	it = 0
	
	# env = PursuitEnv(hunters, preys, field_size, hunter_sight, n_catch, max_steps)
	env = TeamPursuitEnv(hunters, preys, field_size, hunter_sight, {'team_1': ['hunter_1', 'hunter_2']}, n_catch, max_steps, use_layer_obs=True)
	env.seed(RNG_SEED)
	env.set_target('hunter_2', target)
	n_hunters = len(hunters)
	n_preys = len(preys)
	init_pos_hunter = {}
	for hunter in hunters:
		hunter_idx = hunters.index(hunter)
		init_pos_hunter[hunter[0]] = (hunter_idx // n_hunters, hunter_idx % n_hunters)
	init_pos_prey = {}
	for prey in preys:
		prey_idx = preys.index(prey)
		init_pos_prey[prey[0]] = (max(field_size[0] - (prey_idx // n_preys) - 1, 0), max(field_size[1] - (prey_idx % n_preys) - 1, 0))
	env.spawn_hunters(init_pos_hunter)
	env.spawn_preys(init_pos_prey)
	env.update_objectives({'team_1': 'prey_1'})
	obs, *_ = env.reset()
	print('Teams: ', env.teams)
	print('Objectives: ', env.team_objectives)
	
	for i in range(max_steps * 2):
		
		print('Iteration: %d\tStep: %d' % (it + 1, i + 1))
		print(env.field)
		print([(env.agents[key].agent_id, env.agents[key].pos) for key in env.prey_ids])
		actions = []
		for hunter in hunters:
			print(env.agents[hunter[0]].pos)
			action = input('%s action: ' % hunter[0])
			actions += [int(ACTION_MAP[action])]
		for _ in preys:
			actions += [np.random.choice(len(Action))]
		
		print(' '.join([str(Action(action)) for action in actions]))
		obs, rewards, finished, timeout, _ = env.step(actions)
		for idx in range(len(env.hunter_ids)):
			rewards[idx] = env.agents[env.hunter_ids[idx]].get_reward(rewards[idx], env=env)
			for layer in obs[idx]:
				print(layer)
			print('\n')
		print(rewards, finished, timeout)
		# print(env.field)

		if finished:
			print('Finished\n\n')
			obs, *_ = env.reset()
			it += 1


if __name__ == '__main__':
	main()
