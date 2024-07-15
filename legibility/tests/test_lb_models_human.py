#! /usr/bin/env python
import numpy as np
import flax.linen as nn
import yaml
import argparse

from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_algos.dqn import DQNetwork
from itertools import product
from pathlib import Path
from gymnasium.spaces import MultiBinary

np.set_printoptions(precision=5, threshold=10000)

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
KEY_MAP = {'w': 1, 's': 2, 'a': 3, 'd': 4, 'q': 0, 'e': 5}
RNG_SEED = 25456789
TEMPS = [0.1, 0.15, 0.2, 0.25, 0.5, 1.0]
N_CYCLES = 2


def main():

	parser = argparse.ArgumentParser(description='LB-Foraging model train with human inputs')
	parser.add_argument('--seed', type=int, default=RNG_SEED)
	parser.add_argument('--legible', dest='legible', action='store_true')

	args = parser.parse_args()

	n_leg_agents = 1
	n_players = 2
	player_level = 1
	field_size = (8, 8)
	n_foods = 8
	n_foods_spawn = 2
	sight = 8
	max_steps = 600
	food_level = 2
	architecture = "v3"
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'lb_foraging'
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = [tuple(x) for x in config_params['food_locs'][dict_idx]]
			food_confs = [tuple(x) for x in config_params['food_confs'][dict_idx][(n_foods_spawn - 1)]]
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
	
	obj_food = tuple([3, 0])
	env = FoodCOOPLBForaging(n_players, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs, food_locs[1],
							 render_mode=['rgb_array', 'human'], use_encoding=False, agent_center=True, grid_observation=True)
	
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
	legible_dqn_model = DQNetwork(env.action_space[0].n, n_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, use_tracker, cnn_properties=cnn_properties)
	optim_dqn_model = DQNetwork(env.action_space[0].n, n_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, use_tracker, cnn_properties=cnn_properties)
	obs_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	legible_dqn_model.load_model(('food_%dx%d_single_model' % (obj_food[0], obj_food[1])), leg_dir, None, obs_shape, False)
	optim_dqn_model.load_model(('food_%dx%d_single_model' % (obj_food[0], obj_food[1])), optim_dir, None, obs_shape, False)

	rng_gen = np.random.default_rng(RNG_SEED)
	env.set_objective(obj_food)
	env.seed(seed=RNG_SEED)
	env.spawn_food(n_foods_spawn, food_level)
	env.spawn_players()
	print('Food objective is (%d, %d)' % (obj_food[0], obj_food[1]))
	print('Foods spawned: ' + str(obj_food) + ' ' + str(env.food_spawn_pos))
	obs, *_ = env.reset(seed=RNG_SEED)
	# env.spawn_players([1, 1], [(2, 1), (5, 3)])
	env.render()
	finished_runs = 0
	timeout_runs = 0

	for i in range(250):

		print('Iteration: %d' % (i + 1))
		print(env.get_full_env_log())
		done = False
		while not done:
			actions = []
			for a_idx in range(n_players):
				if a_idx > 0:
					if args.legible:
						print('Legible action!')
						online_params = legible_dqn_model.online_state.params
						if use_cnn:
							q_values = legible_dqn_model.q_network.apply(online_params, obs[a_idx].reshape((1, *obs_shape)))[0]
						else:
							q_values = legible_dqn_model.q_network.apply(online_params, obs[a_idx])
					else:
						print('Optimal action!')
						online_params = optim_dqn_model.online_state.params
						if use_cnn:
							q_values = optim_dqn_model.q_network.apply(online_params, obs[a_idx].reshape((1, *obs_shape)))[0]
						else:
							q_values = optim_dqn_model.q_network.apply(online_params, obs[a_idx])

					pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
					pol = pol / pol.sum()
					action = rng_gen.choice(range(env.action_space[0].n), p=pol)
					actions.append(action)

				else:
					valid_action = False
					while not valid_action:
						human_input = input("Action for agent %d:\t" % (a_idx + 1))
						if human_input not in KEY_MAP.keys():
							print('Action %s is invalid, please provide one of the following valid actions: \t' % human_input, str(KEY_MAP.keys()))
						else:
							action = int(KEY_MAP[human_input])
							if action < 6:
								valid_action = True
								actions.append(action)
							else:
								print('Action ID must be between 0 and 5, you gave ID %d' % action)

			print('Actions: ' + ' & '.join([ACTION_MAP[action] for action in actions]))
			next_obs, rewards, finished, timeout, info = env.step(actions)
			print(env.get_env_log())
			print('Rewards: ', str(rewards))
			env.render()
			# input()

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
