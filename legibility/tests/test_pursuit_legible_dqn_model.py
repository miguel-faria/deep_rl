#! /usr/bin/env python

import yaml
import os
import sys
import jax
import numpy as np
import flax.linen as nn

from dl_envs.pursuit.pursuit_env import TargetPursuitEnv, Action
from dl_algos.dqn import DQNetwork
from itertools import product
from pathlib import Path
from gymnasium.spaces import MultiBinary, MultiDiscrete
from typing import List


RNG_SEED = 4072023
TEST_RNG_SEED = 12072023
ACTION_DIM = 5
MAX_EPOCH = 500
PREY_TYPES = {'idle': 0, 'greedy': 1, 'random': 2}


def get_history_entry(obs: np.ndarray, actions: List[int], hunter_ids: List[str]) -> List:
	entry = []
	for hunter in hunter_ids:
		a_idx = hunter_ids.index(hunter)
		state_str = ' '.join([str(x) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def main():
	
	architecture = 'v3'
	gamma = 0.95
	dueling_dqn = True
	use_ddqn = True
	use_cnn = True
	use_render = True
	n_hunters = 3
	n_legible_agents = 1
	n_preys = 4
	n_spawn_preys = 4
	hunter_ids = ['h%d' % i for i in range(1, n_hunters + 1)]
	field_size = (10, 10)
	sight = field_size[0]
	prey_ids = ['p%d' % i for i in range(1, n_preys + 1)]
	prey_type = 'idle'
	n_catch = 2
	catch_reward = 5
	max_steps = 50
	n_cycles = 10
	hunters = []
	preys = []
	for idx in range(n_hunters):
		hunters += [(hunter_ids[idx], 1)]
	for idx in range(n_preys):
		preys += [(prey_ids[idx], PREY_TYPES[prey_type])]
	
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = ('test_pursuit_dqn_%dx%d-field_%d-hunters_%d-catch' % (field_size[0], field_size[1], n_hunters, n_catch)) + '_best'
	optimal_model_path = (models_dir / 'pursuit_single_vdn_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-hunters' % n_hunters)) / ('%s-prey' % prey_type) / 'best'
	legible_model_path = (models_dir / 'pursuit_legible_vdn_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-hunters' % n_hunters)) / ('%s-prey' % prey_type) / 'best'
	
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
	
	print('#############################')
	print('Starting LB Foraging DQN Test')
	print('#############################')
	print('Environment setup')
	rng_gen = np.random.default_rng(TEST_RNG_SEED)
	env = TargetPursuitEnv(hunters, preys, field_size, sight, prey_ids[rng_gen.integers(n_preys)], n_catch, max_steps, use_layer_obs=True, agent_centered=True, catch_reward=catch_reward)
	env.seed(TEST_RNG_SEED)
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	
	print('Setup multi-agent DQN')
	legible_dqn_model = DQNetwork(env.action_space[0].n, n_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties=cnn_properties)
	optim_dqn_model = DQNetwork(env.action_space[0].n, n_layers, nn.relu, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties=cnn_properties)
	obs_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	legible_dqn_model.load_model(('%d-preys_single_model.model' % n_spawn_preys), legible_model_path, None, obs_shape, False)
	optim_dqn_model.load_model(('%d-preys_single_model.model' % n_spawn_preys), optimal_model_path, None, obs_shape, False)
	
	print('Testing trained model')
	np.random.seed(TEST_RNG_SEED)
	for _ in range(n_cycles):
		obs, *_ = env.reset()
		epoch = 0
		history = []
		game_over = False
		if use_render:
			env.render()
		print('Agents: ' + ', '.join(['%s @ (%d, %d)' % (env.agents[hunter].agent_id, *env.agents[hunter].pos) for hunter in env.hunter_ids]))
		print('Preys: ' + ', '.join(['%s @ (%d, %d)' % (env.agents[prey].agent_id, *env.agents[prey].pos) for prey in env.prey_alive_ids]))
		print('Objective prey: %s @ (%d, %d)' % (env.target, *env.agents[env.target].pos))
		while not game_over:
			
			actions = []
			for a_idx in range(env.n_hunters):
				
				if a_idx < n_legible_agents:
					online_params = legible_dqn_model.online_state.params
					# online_params = optim_dqn_model.online_state.params
					
					if use_cnn:
						cnn_obs = obs[a_idx].reshape((1, *obs_shape))
						q_values = legible_dqn_model.q_network.apply(online_params, cnn_obs)[0]
					else:
						q_values = legible_dqn_model.q_network.apply(online_params, obs[a_idx])
					
				else:
					online_params = optim_dqn_model.online_state.params
				
					if use_cnn:
						cnn_obs = obs[a_idx].reshape((1, *obs_shape))
						q_values = optim_dqn_model.q_network.apply(online_params, cnn_obs)[0]
					else:
						q_values = optim_dqn_model.q_network.apply(online_params, obs[a_idx])
				
				pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
				pol = pol / pol.sum()
				action = rng_gen.choice(range(env.action_space[0].n), p=pol)
				# action = jax.device_get(action)
				actions += [action]
				
			for prey_id in env.prey_alive_ids:
				actions += [env.agents[prey_id].act(env)]
			
			actions = np.array(actions)
			print('Actions: ', ' '.join([str(Action(action).name) for action in actions]))
			next_obs, rewards, finished, timeout, infos = env.step(actions)
			print(env.get_env_log())
			obs = next_obs
			epoch += 1
			if use_render:
				env.render()
			input()
			
			if finished or timeout:
				game_over = True
				env.target = prey_ids[rng_gen.integers(n_preys)]
			
			sys.stdout.flush()
		
		print('Epochs needed to finish: %d' % epoch)
	

if __name__ == '__main__':
	main()
