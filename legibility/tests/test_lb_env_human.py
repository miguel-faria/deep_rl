#! /usr/bin/env python
import numpy as np
import flax.linen as nn
import yaml

from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv
from dl_algos.single_model_madqn import SingleModelMADQN
from dl_algos.dqn import DQNetwork
from itertools import product
from pathlib import Path
from gymnasium.spaces import MultiBinary, MultiDiscrete

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
KEY_MAP = {'w': 1, 's': 2, 'a': 3, 'd': 4, 'q': 0, 'e': 5}
RNG_SEED = 123456789


def main():
	
	n_agents = 2
	player_level = 1
	field_size = (8, 8)
	n_foods = 8
	n_foods_spawn = 2
	sight = 8
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
	use_render = True
	architecture = 'v3'
	tensorboard_details = []
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
	
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	optim_dir = (models_dir / 'lb_coop_legible_vdn_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
				 ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / 'best')
	obj_food = food_locs[0]
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs,
	                         use_encoding=True, agent_center=True, grid_observation=use_cnn, use_render=use_render)
	# env = LBForagingEnv(n_agents, player_level, field_size, n_foods, sight, max_steps, True, render_mode=['rgb_array', 'human'], grid_observation=True)
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	agent_action_space = env.action_space[0]
	action_space = MultiDiscrete([agent_action_space.n] * env.n_players)
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	# legible_dqn = SingleModelMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, action_space, obs_space,
	# 							   use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard, tensorboard_details)
	# legible_dqn.load_model(('food_%dx%d' % (obj_food[0], obj_food[1])), optim_dir, None,
	# 					   env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape))

	legible_dqn = DQNetwork(env.action_space[0].n, n_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties)
	legible_dqn.load_model('food_%dx%d_single_model.model' % (obj_food[0], obj_food[1]), optim_dir, None, cnn_shape, False)

	# if food_confs is not None:
	# 	food_conf = food_confs[np.random.choice(range(len(food_confs)))]
	# 	locs = [food for food in food_locs if food != obj_food]
	# 	foods_spawn = [locs[idx] for idx in food_conf]
	# 	env.food_spawn_pos = foods_spawn
	rng_gen = np.random.default_rng(RNG_SEED)
	env.set_objective(obj_food)
	env.seed(seed=123456799)
	env.spawn_food(n_foods_spawn, food_level)
	env.spawn_players()
	print('Food objective is (%d,%d)' % (obj_food[0], obj_food[1]))
	print('Foods spawned: ' + str(obj_food) + ' ' +  str(env.food_spawn_pos))
	obs, *_ = env.reset(seed=123456789)
	print(env.get_full_env_log())
	env.render()
	input()
	
	for i in range(100):

		print('Iteration: %d' % (i + 1))
		actions = []
		for a_idx in range(n_agents):
			if a_idx < 1:
				valid_action = False
				while not valid_action:
					human_input = input("Action for agent %d:\t" % (a_idx + 1))
					action = int(KEY_MAP[human_input])
					if action < 6:
						valid_action = True
						actions.append(action)
					else:
						print('Action ID must be between 0 and 5, you gave ID %d' % action)
			else:
				online_params = legible_dqn.online_state.params
				if legible_dqn.cnn_layer:
					q_values = legible_dqn.q_network.apply(online_params, obs[a_idx].reshape((1, *cnn_shape)))[0]
				else:
					q_values = legible_dqn.q_network.apply(online_params, obs[a_idx])
				pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
				pol = pol / pol.sum()
				print(q_values, pol)
				action = rng_gen.choice(range(env.action_space[0].n), p=pol)
				actions.append(action)

		print('Actions: ' + ' '.join([ACTION_MAP[action] for action in actions]))
		obs, rewards, done, timeout, info = env.step(actions)
		joint_obs = []
		for idx in range(len(obs)):
			joint_obs.append(obs[idx])
		joint_obs = np.array(joint_obs)
		print(obs.shape, joint_obs.shape)
		print(env.get_full_env_log())
		print(rewards)
		if done or timeout:
			obs, *_ = env.reset()
			print(env.get_full_env_log())
		env.render()
		input()


if __name__ == '__main__':
	main()
