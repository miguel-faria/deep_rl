#! /usr/bin/env python
import wandb
import os
import sys
import argparse
import numpy as np
import flax.linen as nn
import gymnasium
import jax
import json
import random
import time
import logging
import yaml

from dl_algos.single_model_madqn import LegibleSingleMADQN
from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_envs.pursuit.pursuit_env import PursuitEnv, TargetPursuitEnv
from dl_envs.pursuit.agents.agent import Agent, AgentType
from pathlib import Path
from itertools import permutations
from typing import List, Tuple
from datetime import datetime

RNG_SEED = 6102023
TEST_RNG_SEED = 4072023
N_TESTS = 100
PREY_TYPES = {'idle': 0, 'greedy': 1, 'random': 2}


def eps_cycle_schedule(schedule_type: str, cycle_nr: int, max_cycles: int, init_eps: float, final_eps: float, decay_rate: float) -> float:
	if schedule_type == 'log':
		return max(init_eps * (1 / (1 + decay_rate * cycle_nr)), final_eps)
	elif schedule_type == 'exp':
		return max(init_eps - decay_rate ** ((max_cycles - 1) / cycle_nr), final_eps)
	else:
		return max(init_eps * (decay_rate ** cycle_nr), final_eps)


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def get_target_seqs(targets: List[str]) -> List[Tuple[str]]:
	if len(targets) > 0:
		return list(permutations(targets, len(targets)))
	else:
		return None


def train_pursuit_legible_dqn(dqn_model: LegibleSingleMADQN, env: PursuitEnv, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float,
							  tau: float, initial_eps: float, final_eps: float, eps_type: str, reward_type: str, rng_seed: int, logger: logging.Logger,
							  cnn_shape: Tuple[int], exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100,
							  tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0, greedy_action: bool = True, sofmax_temp: float = 1.0) -> List:
	rng_gen = np.random.default_rng(rng_seed)
	
	history = []
	# Setup DQNs for training
	obs, *_ = env.reset()
	dqn_model.agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
	
	start_time = time.time()
	sys.stdout.flush()
	start_record_it = cycle * num_iterations
	start_record_epoch = cycle * max_timesteps * env.n_preys
	epoch = start_record_epoch
	avg_loss = []
	
	for it in range(num_iterations):
		if use_render:
			env.render()
		done = False
		episode_rewards = 0
		episode_q_vals = 0
		episode_start = epoch
		episode_history = []
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		while not done:
			
			# interact with environment
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			
			if rng_gen.random() < eps:
				actions = np.array(list(env.action_space.sample()[:env.n_hunters]) + [env.agents[prey_id].act(env) for prey_id in env.prey_alive_ids])
			else:
				actions = []
				for a_idx in range(env.n_hunters):
					if dqn_model.agent_dqn.cnn_layer:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_model.goal].params,
																	   obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
					else:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_model.goal].params, obs[a_idx])
					
					if greedy_action:
						action = q_values.argmax(axis=-1)
					else:
						pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
						pol = pol / pol.sum()
						action = rng_gen.choice(range(env.action_space[0].n), p=pol)
					action = jax.device_get(action)
					episode_q_vals += (float(q_values[int(action)]) / env.n_hunters)
					actions += [action]
				
				for prey_id in env.prey_alive_ids:
					actions += [env.agents[prey_id].act(env)]
				actions = np.array(actions)
			episode_history += [dqn_model.get_history_entry(obs, actions)]
			logger.info(env.get_env_log())
			next_obs, rewards, terminated, timeout, infos = env.step(actions)
			if use_render:
				env.render()
			
			legible_rewards = np.zeros(dqn_model.num_agents)
			live_goals = env.prey_alive_ids
			n_goals = len(env.prey_alive_ids)
			if n_goals > 1:
				for a_idx in range(dqn_model.num_agents):
					act_q_vals = np.zeros(n_goals)
					action = actions[a_idx]
					goal_action_q = 0.0
					for g_idx in range(n_goals):
						if dqn_model.agent_dqn.cnn_layer:
							obs_reshape = obs[a_idx].reshape((1, *obs[a_idx].shape))
							q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[live_goals[g_idx]].params, obs_reshape)[0]
						else:
							q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[live_goals[g_idx]].params, obs[a_idx])
						if dqn_model.goal == live_goals[g_idx]:
							goal_action_q = q_vals[action]
						logger.info(np.exp(dqn_model.beta * (q_vals[action] - q_vals.max()) / sofmax_temp))
						act_q_vals[g_idx] = np.exp(dqn_model.beta * (q_vals[action] - q_vals.max()) / sofmax_temp)
					if reward_type == 'reward':
						legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) * rewards[a_idx]
					elif reward_type == 'q_vals':
						legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) * goal_action_q
					elif reward_type == 'info':
						legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) + rewards[a_idx]
					else:
						legible_rewards[a_idx] = act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()
			else:
				for a_idx in range(dqn_model.n_leg_agents):
					legible_rewards[a_idx] = rewards[a_idx]
			
			if terminated or (timeout and infos['preys_left'] > 0) or ('caught_target' in infos.keys() and infos['caught_target']):
				finished = np.ones(dqn_model.num_agents)
			# env.reset_timestep()
			else:
				finished = np.zeros(dqn_model.num_agents)
			
			logger.info(str(finished) + '\tTimestep: %d' % env.env_timestep)
			
			# store new samples
			if dqn_model.use_vdn:
				dqn_model.replay_buffer.add(obs, next_obs, actions,
											np.hstack((legible_rewards[:dqn_model.n_leg_agents], rewards[dqn_model.n_leg_agents:env.n_hunters])),
											finished[0], [infos])
				episode_rewards += sum(legible_rewards) / dqn_model.n_leg_agents
			else:
				for a_idx in range(env.n_hunters):
					dqn_model.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], legible_rewards[a_idx], finished[a_idx], [])
					episode_rewards += legible_rewards[a_idx] / dqn_model.n_leg_agents
			
			if dqn_model.agent_dqn.use_summary:
				dqn_model.agent_dqn.summary_writer.add_scalar("charts/reward", sum(rewards[:env.n_hunters]) / env.n_hunters, epoch + start_record_epoch)
			obs = next_obs
			
			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					loss = jax.device_get(dqn_model.update_model(batch_size, epoch - start_record_epoch, start_time,
																 tensorboard_frequency, logger, cnn_shape=cnn_shape))
					avg_loss += [loss]
				
				if epoch % target_freq == 0:
					dqn_model.agent_dqn.update_target_model(tau)
			
			epoch += 1
			sys.stdout.flush()
			
			# Check if iteration is over
			if terminated or timeout:
				if dqn_model.write_tensorboard:
					episode_len = epoch - episode_start
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/episodic_q_vals", episode_q_vals / episode_len, epoch)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it + start_record_it)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/iteration", it, it + start_record_it)
					dqn_model.agent_dqn.summary_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), it + start_record_it)
					dqn_model.agent_dqn.summary_writer.add_scalar("losses/td_loss", sum(avg_loss) / max(len(avg_loss), 1), epoch)
				env.reset_init_pos()
				obs, *_ = env.reset()
				done = True
				history += [episode_history]
				episode_rewards = 0
				episode_start = epoch
				avg_loss = []
	
	return history


# noinspection DuplicatedCode
def main():
	parser = argparse.ArgumentParser(description='Train DQN for LB Foraging with fixed foods in environment')
	
	# Multi-agent DQN params
	parser.add_argument('--nagents', dest='n_agents', type=int, required=True, help='Number of agents in the environment')
	parser.add_argument('--architecture', dest='architecture', type=str, required=True, help='DQN architecture to use from the architectures yaml')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	parser.add_argument('--layer-sizes', dest='layer_sizes', type=int, required=True, nargs='+', help='Size of each layer of the DQN\'s neural net')
	
	# Train parameters
	# parser.add_argument('--cycles', dest='n_cycles', type=int, default=0,
	# 					help='Number of training cycles, each cycle spawns the field with a different food items configurations.')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.5, help='Decay rate for the exploration update')
	parser.add_argument('--cycle-eps-decay', dest='cycle_eps_decay', type=float, required=False, default=0.95, help='Decay rate for the exploration update')
	parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default='log', choices=['linear', 'exp', 'log', 'epoch'],
						help='Type of exploration rate update to use: linear, exponential (exp), logarithmic (log), epoch based (epoch)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
						help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	parser.add_argument('--restart', dest='restart_train', action='store_true',
						help='Flag that signals that train is suppose to restart from a previously saved point.')
	parser.add_argument('--restart-info', dest='restart_info', type=str, nargs='+', required=False, default=None,
						help='List with the info required to recover previously saved model and restart from same point: '
							 '<model_dirname: str> <model_filename: str> <last_cycle: int> Use only in combination with --restart option')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	parser.add_argument('--train-targets', dest='train_targets', type=str, nargs='+', required=False, default=None,
						help='List with the prey ids to train to catch')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
						help='Directory to store trained models and load optimal models, if left blank stored in default location')
	
	# Environment parameters
	parser.add_argument('--hunter-ids', dest='hunter_ids', type=str, nargs='+', required=True, help='List with the hunter ids in the environment')
	parser.add_argument('--prey-ids', dest='prey_ids', type=str, nargs='+', required=True, help='List with the prey ids in the environment')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--hunter-classes', dest='hunter_class', type=int, required=True, help='Class of agent to use for the hunters')
	parser.add_argument('--n-hunters-catch', dest='require_catch', type=int, required=True, help='Minimum number of hunters required to catch a prey')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag that signals the use of the field render while training')
	
	args = parser.parse_args()
	# DQN args
	n_agents = args.n_agents
	architecture = args.architecture
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_cnn = args.use_cnn
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	layer_sizes = args.layer_sizes
	
	# Train args
	n_iterations = args.n_iterations
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	learn_rate = args.learn_rate
	target_update_rate = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tensorboard_freq = args.tensorboard_freq
	debug = args.debug
	cycle_decay = args.cycle_eps_decay
	
	# LB-Foraging environment args
	hunter_ids = args.hunter_ids
	prey_ids = args.prey_ids
	field_lengths = args.field_lengths
	max_steps = args.max_steps
	use_render = args.use_render
	require_catch = args.require_catch
	hunter_class = args.hunter_class
	tags = args.tags if args.tags is not None else ''
	
	hunters = []
	preys = []
	n_hunters = len(hunter_ids)
	n_preys = len(prey_ids)
	for idx in range(n_hunters):
		hunters += [(hunter_ids[idx], hunter_class)]
	for idx in range(n_preys):
		preys += [(prey_ids[idx], PREY_TYPES['random'])]
	
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.fraction
	if not use_gpu:
		jax.config.update('jax_platform_name', 'cpu')
	
	# logger.info(gamma, initial_eps, final_eps, eps_decay, eps_type, warmup, learn_rate, target_learn_rate)
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
	
	now = datetime.now()
	log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs'
	models_dir = args.models_dir if args.models_dir != '' else Path(__file__).parent.absolute().parent.absolute() / 'models'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	log_filename = (('train_pursuit_single_dqn_%dx%d-field_%d-hunters_%d-preys' % (field_size[0], field_size[1], n_hunters, n_preys)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / 'pursuit_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-hunters' % n_hunters) /
				  ('%d-preys' % n_preys) / now.strftime("%Y%m%d-%H%M%S"))
	
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
	
	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)
	
	logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
						level=logging.INFO, encoding='utf-8')
	logger = logging.getLogger('INFO')
	err_logger = logging.getLogger('ERROR')
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
	err_logger.addHandler(handler)
	Path.mkdir(model_path, parents=True, exist_ok=True)
	preys_permutations = list(permutations(prey_ids))
	
	#####################
	## Training Models ##
	#####################
	
	wandb.init(project='pursuit-legible', entity='miguel-faria',
			   config={
				   "field": "%dx%d" % (field_size[0], field_size[1]),
				   "agents": n_agents,
				   "preys": n_preys,
				   "hunters": n_hunters,
				   "online_learing_rate": learn_rate,
				   "target_learning_rate": target_update_rate,
				   "discount": gamma,
				   "eps_decay": eps_type,
				   "dqn_architecture": architecture,
				   "iterations": n_iterations,
				   "tags": tags
			   },
			   dir=tensorboard_details[0],
			   name=('%ssingle-l%dx%d-%dh-%dp-%s-' % ('vdn-' if use_vdn else 'independent-', field_size[0], field_size[1], n_hunters, n_preys, prey_type) +
					 now.strftime("%Y%m%d-%H%M%S")),
			   sync_tensorboard=True)
	logger.info('##########################')
	logger.info('Starting Pursuit DQN Train')
	logger.info('##########################')
	n_cycles = n_preys
	logger.info('Number of cycles: %d' % n_cycles)
	logger.info('Number of iterations per cycle: %d' % n_iterations)
	logger.info('Environment setup')
	env = TargetPursuitEnv(hunters, preys, field_size, sight, prey_ids, require_catch, max_steps, use_layer_obs=True)
	# env = PursuitEnv(hunters, preys, field_size, sight, require_catch, max_steps, use_layer_obs=True)
	logger.info('Setup multi-agent DQN')
	dqn_model = LegibleSingleMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, env.observation_space[0],
								 use_gpu, dueling_dqn, use_ddqn, use_cnn, False, use_tensorboard,
								 tensorboard_details + ['%dh-%dp-%dc' % (n_hunters, n_preys, require_catch)])
	random.seed(RNG_SEED)
	prey_lists = [random.choice(preys_permutations) for _ in range(n_cycles)]
	logger.info('Starting training')
	for cycle in range(n_cycles):
		logger.info('Cycle %d of %d' % (cycle + 1, n_cycles))
		env.seed(RNG_SEED)
		sys.stdout.flush()
		if cycle == 0:
			cycle_init_eps = initial_eps
		else:
			cycle_init_eps = eps_cycle_schedule('exp', cycle, n_cycles, initial_eps, final_eps, cycle_decay)
		env.target = prey_lists[cycle] if isinstance(prey_lists[cycle], list) else list(prey_lists[cycle])
		logger.info('Starting exploration: %d with decay of %f' % (cycle_init_eps, eps_decay))
		history = train_pursuit_legible_dqn(dqn_model, env, n_iterations, max_steps * n_iterations, batch_size, learn_rate, target_update_rate, cycle_init_eps,
											final_eps, eps_type, RNG_SEED, logger, eps_decay, warmup, train_freq, target_freq, tensorboard_freq, use_render, cycle)
		
		if debug:
			logger.debug('Saving cycle iteration history')
			json_path = model_path / 'all_preys_history_centralized.json'
			with open(json_path, 'a') as json_file:
				json_file.write(json.dumps({('cycle_%d' % (cycle + 1)): history}))
	
	logger.info('Saving final model')
	dqn_model.save_model('all_preys', model_path, logger)
	sys.stdout.flush()
	
	####################
	## Testing Model ##
	####################
	env = TargetPursuitEnv(hunters, preys, field_size, sight, prey_ids, require_catch, max_steps, use_layer_obs=True)
	env.seed(TEST_RNG_SEED)
	np.random.seed(TEST_RNG_SEED)
	random.seed(TEST_RNG_SEED)
	failed_history = []
	tests_passed = 0
	testing_prey_lists = [random.choice(preys_permutations) for _ in range(N_TESTS)]
	for n_test in range(N_TESTS):
		env.reset_init_pos()
		obs, *_ = env.reset()
		logger.info('Test number %d' % (n_test + 1))
		logger.info('Prey locations: ' + ', '.join(['(%d, %d)' % env.agents[prey_id].pos for prey_id in env.prey_alive_ids]))
		logger.info('Agent positions: ' + ', '.join(['(%d, %d)' % env.agents[hunter_id].pos for hunter_id in env.hunter_ids]))
		logger.info('Testing sequence: ' + ', '.join(testing_prey_lists[n_test]))
		obs, *_ = env.reset()
		epoch = 0
		agent_reward = [0, 0]
		test_history = []
		game_over = False
		finished = False
		timeout = False
		env.target = testing_prey_lists[n_test]
		while not game_over:
			
			actions = []
			for h_idx in range(n_hunters):
				if dqn_model.agent_dqn.cnn_layer:
					q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params,
																   obs[h_idx].reshape((1, *obs[h_idx].shape)))[0]
				else:
					q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, obs[h_idx])
				action = q_values.argmax(axis=-1)
				action = jax.device_get(action)
				actions += [action]
			for prey_id in env.prey_alive_ids:
				actions += [env.agents[prey_id].act(env)]
			actions = np.array(actions)
			next_obs, rewards, finished, timeout, infos = env.step(actions)
			test_history += [get_history_entry(env.make_array_obs(), actions, dqn_model.num_agents)]
			obs = next_obs
			
			if finished or timeout:
				game_over = True
			
			sys.stdout.flush()
			epoch += 1
		
		if finished:
			tests_passed += 1
			logger.info('Test %d finished in success' % (n_test + 1))
			logger.info('Number of epochs: %d' % epoch)
			logger.info('Accumulated reward:\n\t- agent 1: %.2f\n\t- agent 2: %.2f' % (agent_reward[0], agent_reward[1]))
			logger.info('Average reward:\n\t- agent 1: %.2f\n\t- agent 2: %.2f' % (agent_reward[0] / epoch, agent_reward[1] / epoch))
		if timeout:
			failed_history += [test_history]
			logger.info('Test %d timed out' % (n_test + 1))
	
	logger.info('Passed %d tests out of %d' % (tests_passed, N_TESTS))
	logger.info('Failed tests history:')
	logger.info(failed_history)


if __name__ == '__main__':
	main()

