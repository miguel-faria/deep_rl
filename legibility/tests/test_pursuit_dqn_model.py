#! /usr/bin/env python

import argparse
import os
import sys
import jax
import numpy as np
import flax.linen as nn

from dl_algos.multi_model_madqn import MultiAgentDQN
from dl_envs.pursuit.pursuit_env import PursuitEnv, Action
from pathlib import Path
from gymnasium.spaces.multi_discrete import MultiDiscrete
from typing import List


RNG_SEED = 4072023
TEST_RNG_SEED = 12072023
ACTION_DIM = 5
MAX_EPOCH = 500


def get_history_entry(obs: np.ndarray, actions: List[int], hunter_ids: List[str]) -> List:
	entry = []
	for hunter in hunter_ids:
		a_idx = hunter_ids.index(hunter)
		state_str = ' '.join([str(x) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def main():
	parser = argparse.ArgumentParser(description='Test DQN model for for pursuit environment.')
	
	# Multi-agent DQN params
	parser.add_argument('--nlayers', dest='n_layers', type=int, required=True, help='Number of layers for the neural net in the DQN')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	
	# Environment parameters
	parser.add_argument('--hunters', dest='hunters', type=str, nargs='+', required=True, help='IDs of hunters in the environment')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--preys', dest='preys', type=str, nargs='+', required=True, help='IDs of preys in the environment')
	parser.add_argument('--prey-type', dest='prey_type', type=str, required=True, help='Type of prey agent to use')
	parser.add_argument('--n-catch', dest='n_catch', type=int, required=True, help='Number of hunters that have to surround the prey to catch it')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	
	args = parser.parse_args()
	n_layers = args.n_layers
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	tensorboard_freq = args.tensorboard_freq
	hunters = args.hunters
	field_lengths = args.field_lengths
	preys = args.preys
	prey_type = args.prey_type
	n_catch = args.n_catch
	max_steps = args.max_steps
	
	os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
	
	field_dims = len(field_lengths)
	if 2 >= field_dims > 0:
		if field_dims == 1:
			field_size = (field_lengths[0], field_lengths[0])
			sight = field_lengths[0]
		else:
			field_size = (field_lengths[0], field_lengths[1])
			sight = max(field_lengths[0], field_lengths[1])
	else:
		print('[ARGS ERROR] Field size must either be composed of only 1 or 2 arguments; %d were given. Exiting program' % field_dims)
		return
	
	n_hunters = len(hunters)
	n_preys = len(preys)
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = ('test_pursuit_dqn_%dx%d-field_%d-hunters_%d-catch' % (field_size[0], field_size[1], n_hunters, n_catch)) + '_best'
	model_path = (models_dir / 'pursuit_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-hunters' % n_hunters)) / ('%s' % prey_type) / 'best'
	
	sys.stdout = open(log_dir / (log_filename + '_log.txt'), 'a')
	sys.stderr = open(log_dir / (log_filename + '_err.txt'), 'w')
	
	print('#############################')
	print('Starting LB Foraging DQN Test')
	print('#############################')
	print('Environment setup')
	env = PursuitEnv(hunters, preys, field_size, n_hunters, n_catch, max_steps)
	env.seed(RNG_SEED)
	rng_gen = np.random.default_rng(RNG_SEED)
	
	print('Setup multi-agent DQN')
	obs_dims = [field_size[0], field_size[1], 2, 2, n_hunters + 1] * (n_hunters + n_preys)
	agents_dqns = MultiAgentDQN(n_hunters, hunters, len(Action), n_layers, nn.relu, layer_sizes, buffer_size, gamma, MultiDiscrete(obs_dims),
								use_gpu, dueling_dqn, use_ddqn, False, use_tensorboard, tensorboard_details)
	
	agents_dqns.load_models(('%d-catch' % n_catch), model_path)
	
	print('Testing trained model')
	env.seed(TEST_RNG_SEED)
	rng_gen = np.random.default_rng(TEST_RNG_SEED)
	np.random.seed(TEST_RNG_SEED)
	init_pos_hunter = {'hunter_1': (0, 5), 'hunter_2': (7, 0)}
	# for hunter in hunters:
	# 	hunter_idx = hunters.index(hunter)
	# 	init_pos_hunter[hunter] = (hunter_idx // n_hunters, hunter_idx % n_hunters)
	init_pos_prey = {'prey_1': (0, 9)}
	# for prey in preys:
	# 	prey_idx = preys.index(prey)
	# 	init_pos_prey[prey] = (max(field_size[0] - (prey_idx // n_preys) - 1, 0), max(field_size[1] - (prey_idx % n_preys) - 1, 0))
	env.spawn_hunters(init_pos_hunter)
	env.spawn_preys(init_pos_prey)
	obs, *_ = env.reset()
	epoch = 0
	history = []
	game_over = False
	print(env.field)
	while not game_over:
		
		actions = []
		for a_id in agents_dqns.agent_ids:
			agent_dqn = agents_dqns.agent_dqns[a_id]
			a_idx = agents_dqns.agent_ids.index(a_id)
			q_values = agent_dqn.q_network.apply(agent_dqn.online_state.params, obs[a_idx])
			action = q_values.argmax(axis=-1)
			action = jax.device_get(action)
			actions += [action]
		actions = np.array(actions)
		print(' '.join([str(Action(action)) for action in actions]))
		next_obs, rewards, finished, timeout, infos = env.step(actions)
		history += [get_history_entry(obs, actions, hunters)]
		obs = next_obs
		print(env.field)
		
		if finished or epoch >= max_steps:
			game_over = True
		
		sys.stdout.flush()
		epoch += 1
	
	print('Epochs needed to finish: %d' % epoch)
	print('Test history:')
	print(history)
	

if __name__ == '__main__':
	main()
