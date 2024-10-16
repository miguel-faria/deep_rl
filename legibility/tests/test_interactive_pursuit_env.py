#! /usr/bin/env python
import numpy as np

from dl_envs.pursuit.pursuit_env import PursuitEnv, Action, TargetPursuitEnv
from typing import List

RNG_SEED = 12072023
ACTION_MAP = {'w': Action.UP, 's': Action.DOWN, 'a': Action.LEFT, 'd': Action.RIGHT, 'q': Action.STAY}


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx][0]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry



def main():
	
	# hunters = ['hunter_1', 'hunter_2', 'hunter_3', 'hunter_4']
	# preys = ['prey_1', 'prey_2']
	hunter_ids = ['hunter_1', 'hunter_2']
	prey_ids = ['prey_1', 'prey_2']
	field_size = (10, 10)
	hunter_sight = 10
	max_steps = 50
	n_catch = len(hunter_ids)
	it = 0
	hunters = []
	preys = []
	n_hunters = len(hunter_ids)
	n_preys = len(prey_ids)
	for idx in range(n_hunters):
		hunters += [(hunter_ids[idx], 1)]
	for idx in range(n_preys):
		preys += [(prey_ids[idx], 1)]
	
	# env = PursuitEnv(hunters, preys, field_size, hunter_sight, n_catch, max_steps)
	env = TargetPursuitEnv(hunters, preys, field_size, hunter_sight, prey_ids[0], n_catch, max_steps, use_layer_obs=True, agent_centered=True)
	env.seed(RNG_SEED)
	free_pos = [(row, col) for row in range(field_size[0]) for col in range(field_size[1])]
	init_pos_hunter = {}
	for hunter in hunters:
		n_free_pos = len(free_pos)
		hunter_pos = np.random.choice(n_free_pos)
		init_pos_hunter[hunter[0]] = free_pos[hunter_pos]
		free_pos.pop(hunter_pos)
	init_pos_prey = {}
	for prey in preys:
		n_free_pos = len(free_pos)
		prey_pos = np.random.choice(n_free_pos)
		init_pos_prey[prey[0]] = free_pos[prey_pos]
		free_pos.pop(prey_pos)
	env.spawn_hunters(init_pos_hunter)
	env.spawn_preys(init_pos_prey)
	env.target = ['prey_2', 'prey_1']
	obs, *_ = env.reset()
	
	for i in range(max_steps * 2):
		
		print('Iteration: %d\tStep: %d' % (it + 1, i + 1))
		print(env.field)
		print([(env.agents[key].agent_id, env.agents[key].pos) for key in env.hunter_ids])
		print([(env.agents[key].agent_id, env.agents[key].pos) for key in env.prey_alive_ids])
		print('Target prey: %s' % env.target[env.target_idx])
		actions = []
		for hunter in hunters:
			print(env.agents[hunter[0]].pos)
			action = input('%s action: ' % hunter[0])
			actions += [int(ACTION_MAP[action])]
		for _ in preys:
			actions += [np.random.choice(len(Action))]
		
		print(' '.join([str(Action(action)) for action in actions]))
		obs, rewards, finished, timeout, info = env.step(actions)
		print(get_history_entry(env.make_array_obs(), actions, n_hunters))
		for idx in range(len(env.hunter_ids)):
			rewards[idx] = env.agents[env.hunter_ids[idx]].get_reward(rewards[idx], env=env)
			print(str(obs[idx][-1]))
			if 'real_obs' in info.keys() and info['real_obs'] is not None:
				print(str(info['real_obs'][idx][-1]))
		print(rewards, finished, timeout)
		# print(str(info))
		print(env.field)
		if 'real_obs' in info.keys() and info['real_obs'] is not None:
			obs = info['real_obs']

		if finished:
			print('Finished\n\n')
			obs, *_ = env.reset()
			it += 1


if __name__ == '__main__':
	main()
