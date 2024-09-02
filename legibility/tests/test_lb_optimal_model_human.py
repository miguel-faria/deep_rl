#! /usr/bin/env python
import numpy as np
import flax.linen as nn
import yaml

from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_algos.dqn import DQNetwork
from itertools import product
from pathlib import Path
from gymnasium.spaces import MultiBinary
from typing import List, Tuple

np.set_printoptions(precision=5, threshold=10000)

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
KEY_MAP = {'w': 1, 's': 2, 'a': 3, 'd': 4, 'q': 0, 'e': 5}
RNG_SEED = 20240729
TEMPS = [0.1, 0.15, 0.2, 0.25, 0.5, 1.0]
HUMAN_CONTROLLER = 1


def main():
	
	n_leg_agents = 1
	n_players = 2
	player_level = 1
	field_size = (8, 8)
	n_foods = 8
	n_foods_spawn = 6
	sight = 8
	max_steps = 5000
	food_level = 2
	architecture = "v3"
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'lb_foraging'
	
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = config_params['food_locs'][dict_idx]
			food_confs = config_params['food_confs'][dict_idx][(n_foods_spawn - 1)]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	
	with open(data_dir / 'configs' / 'q_network_architectures.yaml') as architecture_file:
		arch_data = yaml.safe_load(architecture_file)
		if architecture in arch_data.keys():
			n_layers = arch_data[architecture]['n_layers']
			layer_sizes = arch_data[architecture]['layer_sizes']
			n_conv_layers = arch_data[architecture]['n_cnn_layers']
			cnn_size = arch_data[architecture]['cnn_size']
			cnn_kernel = [tuple(elem) for elem in arch_data[architecture]['cnn_kernel']]
			pool_window = [tuple(elem) for elem in arch_data[architecture]['pool_window']]
			cnn_properties = [n_conv_layers, cnn_size, cnn_kernel, pool_window]
	
	leg_dir = (models_dir / 'lb_coop_legible_vdn_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_players) /
	           ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / 'best')
	optim_dir = (models_dir / 'lb_coop_single_vdn_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_players) /
	             ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / 'best')
	
	obj_food = [5, 4]
	env = FoodCOOPLBForaging(n_players, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs, use_render=False,
	                         render_mode=['rgb_array', 'human'], use_encoding=True, agent_center=True, grid_observation=True)
	
	# Get optimal models
	gamma = 0.95
	dueling_dqn = True
	use_ddqn = True
	use_cnn = True
	use_tracker = False
	goals = [str(loc) for loc in food_locs]
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	legible_dqn_model = DQNetwork(env.action_space[0].n, n_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties=cnn_properties)
	optim_dqn_model = DQNetwork(env.action_space[0].n, n_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties=cnn_properties)
	obs_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	legible_dqn_model.load_model(('food_%dx%d_single_model.model' % (obj_food[0], obj_food[1])), leg_dir, None, obs_shape, False)
	optim_dqn_model.load_model(('food_%dx%d_single_model.model' % (obj_food[0], obj_food[1])), optim_dir, None, obs_shape, False)
	
	rng_gen = np.random.default_rng(RNG_SEED)
	# env.seed(seed=RNG_SEED)
	env.food_spawn_pos = [[0, 2], [1, 7], [2, 7], [3, 1], [6, 6]]
	env.n_food_spawn = n_foods_spawn
	env.set_objective(obj_food)
	env.spawn_players()
	env.spawn_food(n_foods_spawn, food_level)
	print('Food objective is (%d, %d)' % (obj_food[0], obj_food[1]))
	print('Foods spawned: ' + str(obj_food) + ' ' +  str(env.food_spawn_pos))
	print(env.get_full_env_log())
	obs, *_ = env.reset()
	print(env.get_full_env_log())
	env.render()
	finished_runs = 0
	timeout_runs = 0
	
	for layer in obs[0]:
		for line in layer:
			print(line)
		print('\n')
	
	for i in range(250):
		
		print('Iteration: %d' % (i + 1))
		print(env.get_full_env_log())
		input()
		done = False
		while not done:
			actions = []
			for a_idx in range(n_players):
				# if (a_idx + 1) == HUMAN_CONTROLLER:
				# 	valid_action = False
				# 	while not valid_action:
				# 		human_input = input("Action for agent %d:\t" % (a_idx + 1))
				# 		try:
				# 			action = int(KEY_MAP[human_input])
				# 			if action < 6:
				# 				valid_action = True
				# 				actions.append(action)
				# 			else:
				# 				print('Action ID must be between 0 and 5, you gave ID %d' % action)
				# 		except KeyError as e:
				# 			print('Key error caught: %s' % str(e))
				# else:
				online_params = optim_dqn_model.online_state.params
				if use_cnn:
					q_values = optim_dqn_model.q_network.apply(online_params, obs[a_idx].reshape((1, *obs_shape)))[0]
					leg_q_values = legible_dqn_model.q_network.apply(legible_dqn_model.online_state.params, obs[a_idx].reshape((1, *obs_shape)))[0]
				else:
					q_values = optim_dqn_model.q_network.apply(online_params, obs[a_idx])
					leg_q_values = legible_dqn_model.q_network.apply(legible_dqn_model.online_state.params, obs[a_idx])
				
				print(q_values, q_values - q_values.max(), leg_q_values, leg_q_values - leg_q_values.max())
				pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
				pol = pol / pol.sum()
				action = rng_gen.choice(range(env.action_space[0].n), p=pol)
				actions.append(action)
			
			print(env.get_env_log())
			print('Actions: ' + ' & '.join([ACTION_MAP[action] for action in actions]))
			next_obs, rewards, finished, timeout, info = env.step(actions)
			print(env.get_env_log())
			# print('Rewards: ', str(rewards))
			env.render()
			input()
			
			if finished or timeout:
				if finished:
					print('Result: Finished!!')
					finished_runs += 1
				else:
					print('Result: Timeout!!')
					timeout_runs += 1
				env.food_spawn_pos = None
				obs, *_ = env.reset()
				done = True
				env.render()
			
			obs = next_obs
	
	print('Finished %d out of 250 runs.\tTimeout %d out of 250 runs.' % (finished_runs, timeout_runs))


if __name__ == '__main__':
	main()
