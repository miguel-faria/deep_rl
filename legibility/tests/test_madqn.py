#! /usr/bin/env python

import argparse
import gym
import numpy as np
import flax.linen as nn

from dl_algos.multi_model_madqn import MultiAgentDQN
from pettingzoo.sisl import pursuit_v4
from pathlib import Path

RNG_SEED = 13042023


def test_madqn():
	parser = argparse.ArgumentParser(description='Test of multi-agent DQN with JAX using PettingZoo pursuit v4')
	
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
	# parser.add_argument('--', dest='', type=, required=, help='')

	args = parser.parse_args()
	env = pursuit_v4.env(max_cycles=500, x_size=16, y_size=16, shared_reward=True, n_evaders=args.n_agents*4,
						 n_pursuers=args.n_agents, obs_range=7, n_catch=2, freeze_evaders=False, tag_reward=0.01,
						 catch_reward=5.0, urgency_reward=-0.1, surround=True, constraint_window=1.0)
	
	agents_dqns = MultiAgentDQN(args.n_agents, args.agent_ids, env._action_space(env.possible_agents[1]).n, args.n_layers, nn.relu, args.layer_sizes,
								args.buffer_size, args.gamma, env._observation_space(env.possible_agents[1]), args.use_gpu, True, args.use_tensorboard,
								args.tensorboard_details)
	
	agents_dqns.train_dqns(env, args.n_steps, args.batch_size, args.learn_rate, args.target_learn_rate, args.initial_eps, args.final_eps, args.eps_type,
						   RNG_SEED, args.eps_decay, args.warmup_steps, args.train_freq, args.target_freq, args.tensorboard_freq)

	agents_dqns.save_models('test_madqn', Path(__file__).parent.absolute().parent.absolute() / 'models')


if __name__ == '__main__':
	test_madqn()
