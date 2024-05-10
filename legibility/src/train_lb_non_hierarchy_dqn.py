#! /usr/bin/env python

import sys

import argparse
import numpy as np
import flax.linen as nn
import yaml
import jax

from dl_algos.multi_model_madqn import MultiAgentDQN
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from pathlib import Path
from gym.spaces.multi_discrete import MultiDiscrete
from itertools import product

RNG_SEED = 13042023


def main():
	parser = argparse.ArgumentParser(description='Train DQN for LB Foraging with fixed foods in environment')
	
	# Multi-agent DQN params
	parser.add_argument('--nagents', dest='n_agents', type=int, required=True, help='Number of agents in the environment')
	parser.add_argument('--nlayers', dest='n_layers', type=int, required=True, help='Number of layers for the neural net in the DQN')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	parser.add_argument('--agent-ids', dest='agent_ids', type=str, required=True, nargs='+', help='ID for each agent in the environment')
	
	# Train parameters
	parser.add_argument('--cycles', dest='n_cycles', type=int, required=True,
						help='Number of training cycles, each cycle spawns the field with a different number of food items.')
	parser.add_argument('--steps', dest='n_steps', type=int, required=True,
						help='Number of epochs to run training (negative value signals run until convergence)')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.95, help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
						help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	
	# Environment parameters
	parser.add_argument('--player-level', dest='player_level', type=int, required=True, help='Level of the agents collecting food')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--n-food', dest='n_foods', type=int, required=True, help='Number of food items in the field')
	parser.add_argument('--food-level', dest='food_level', type=int, required=True, help='Level of the food items')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	
	args = parser.parse_args()
	n_agents = args.n_agents
	n_layers = args.n_layers
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	agent_ids = args.agent_ids
	n_cycles = args.n_cycles
	n_steps = args.n_steps
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	learn_rate = args.learn_rate
	target_learn_rate = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	player_level = args.player_level
	field_lengths = args.field_lengths
	n_foods = args.n_foods
	food_level = args.food_level
	max_steps = args.max_steps
	
	if not use_gpu:
		jax.config.update('jax_platform_name', 'cpu')
	
	# print(gamma, initial_eps, final_eps, eps_decay, eps_type, warmup, learn_rate, target_learn_rate)
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
	
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	log_filename = ('train_lb_non_hierarchy_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level' % (field_size[0], field_size[1], n_agents, n_foods, food_level))
	model_path = (models_dir / 'lb_non_hierarchy_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_agents) /
				  ('%d-foods_%d-food-level' % (n_foods, food_level)))
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1]) + '_food_locs_non_hierarchy'
		if dict_idx in config_params.keys():
			foods = config_params[dict_idx]
			food_locs = []
			food_lvls = []
			for food in foods:
				food_locs += [food[:2]]
				food_lvls += [food[2]]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
			food_lvls = [np.random.randint(1, n_agents) for _ in range(len(food_locs))]
	
	sys.stdout = open(log_dir / (log_filename + '_log.txt'), 'a')
	sys.stderr = open(log_dir / (log_filename + '_err.txt'), 'w')
	
	print('##############################')
	print('Starting LB Foraging DQN Train')
	print('##############################')
	print('Environment setup')
	env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, False, food_level, food_locs)
	env.seed(RNG_SEED)
	rng_gen = np.random.default_rng(RNG_SEED)
	
	print('Starting training for different food locations')
	for loc in food_locs:
		print('Training for location: %dx%d' % (loc[0], loc[1]))
		env.obj_food = loc
		print('Setup multi-agent DQN')
		agents_dqns = MultiAgentDQN(n_agents, agent_ids, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma,
									MultiDiscrete(env.observation_space[0].high - env.observation_space[0].low), use_gpu, False, use_tensorboard,
									tensorboard_details)
		for cycle in range(n_cycles):
			print('Cycle %d of %d' % (cycle+1, n_cycles))
			if cycle == 0:
				foods_spawn = n_foods
			else:
				foods_spawn = rng_gen.choice(range(n_foods))
			env.spawn_players(player_level)
			env.spawn_food(foods_spawn, food_level)
			print('Starting train')
			sys.stdout.flush()
			agents_dqns.train_dqns(env, n_steps, batch_size, learn_rate, target_learn_rate, initial_eps, final_eps, eps_type,
								   RNG_SEED, log_filename + '_log.txt', eps_decay, warmup, train_freq, target_freq, tensorboard_freq)
	
		Path.mkdir(model_path, parents=True, exist_ok=True)
		agents_dqns.save_models(('food_%dx%d' % (loc[0], loc[1])), model_path)
		sys.stdout.flush()
	

if __name__ == '__main__':
	main()
	