#! /usr/bin/env python
import wandb
import os
import sys
import argparse
import numpy as np
import flax.linen as nn
import gymnasium
import jax
import random
import time
import logging
import yaml
import gc

from dl_algos.single_model_madqn import SingleModelMADQN
from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_envs.pursuit.pursuit_env import PursuitEnv, TargetPursuitEnv
from memory_profiler import profile
from pathlib import Path
from itertools import permutations
from typing import List, Tuple, Union
from datetime import datetime


RNG_SEED = 6102023
TEST_RNG_SEED = 4072023
N_TESTS = 100
PREY_TYPES = {'idle': 0, 'greedy': 1, 'random': 2}


def input_callback(env: Union[PursuitEnv, TargetPursuitEnv], stop_flag: bool):
	try:
		while not stop_flag:
			command = input('Interactive commands:\n\trender - display renderization of the interaction\n\tstop_render - stops the renderization\nCommand: ')
			if command == 'render':
				env.use_render = True
			elif command == 'stop_render':
				if env.use_render:
					env.close_render()
					env.use_render = False
	
	except KeyboardInterrupt as ki:
		return


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
		state_str = ' '.join([str(int(x)) for x in obs[a_idx][0]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def get_target_seqs(targets: List[str]) -> List[Tuple[str]]:
	
	if len(targets) > 0:
		return list(permutations(targets, len(targets)))
	else:
		return None


@profile
def train_pursuit_dqn(dqn_model: SingleModelMADQN, env: PursuitEnv, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float,
					  tau: float, initial_eps: float, final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, cnn_shape: Tuple[int],
					  exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1,
					  use_render: bool = False, cycle: int = 0, greedy_action: bool = True) -> List:
	rng_gen = np.random.default_rng(rng_seed)
	
	history = []
	# Setup DQNs for training
	obs, *_ = env.reset()
	if not dqn_model.agent_dqn.dqn_initialized:
		logger.info('Initializing network')
		if dqn_model.agent_dqn.cnn_layer:
			cnn_obs = obs[0].reshape((1, *cnn_shape))
			dqn_model.agent_dqn.init_network_states(rng_seed, cnn_obs, optim_learn_rate)
		else:
			dqn_model.agent_dqn.init_network_states(rng_seed, obs[0], optim_learn_rate)
	dqn_model.replay_buffer.reset_seed()
	
	start_time = time.time()
	sys.stdout.flush()
	start_record_it = cycle * num_iterations
	start_record_epoch = cycle * max_timesteps * env.n_preys
	epoch = start_record_epoch
	eps = initial_eps
	
	for it in range(num_iterations):
		if use_render and eps <= final_eps:
			env.render()
		done = False
		episode_rewards = 0
		episode_q_vals = 0
		episode_start = epoch
		episode_history = []
		while not done:
			
			# interact with environment
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			
			if rng_gen.random() < eps:
				actions = np.hstack((env.action_space.sample()[:env.n_hunters], np.array([env.agents[prey].act(env) for prey in env.prey_alive_ids])))
				# actions = env.action_space.sample()
			else:
				actions = []
				for a_idx in range(env.n_hunters):
					if dqn_model.agent_dqn.cnn_layer:
						cnn_obs = obs[a_idx].reshape((1, *cnn_shape))
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, cnn_obs)[0]
					else:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, obs[a_idx])
					
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
			if use_render and eps <= final_eps:
				env.render()
			
			if len(rewards) == 1:
				rewards = np.array([rewards] * dqn_model.num_agents)
			
			if terminated or ('caught_target' in infos.keys() and infos['caught_target']):
				finished = np.ones(dqn_model.num_agents, dtype=np.int32)
			else:
				finished = np.zeros(dqn_model.num_agents, dtype=np.int32)
			
			# store new samples
			if dqn_model.use_vdn:
				hunter_actions = actions[:env.n_hunters]
				dqn_model.replay_buffer.add(obs, next_obs, hunter_actions, rewards[:env.n_hunters], finished[0], [])
			else:
				for a_idx in range(env.n_hunters):
					dqn_model.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], rewards[a_idx], finished[a_idx], [])
			episode_rewards += (sum(rewards[:env.n_hunters]) / env.n_hunters)
			if dqn_model.agent_dqn.use_summary:
				dqn_model.agent_dqn.summary_writer.add_scalar("charts/reward", sum(rewards[:env.n_hunters]) / env.n_hunters, epoch + start_record_epoch)
			obs = next_obs
			
			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					loss = jax.device_get(dqn_model.update_model(batch_size, epoch - start_record_epoch, start_time,
																 tensorboard_frequency, logger, cnn_shape=cnn_shape))
					if dqn_model.write_tensorboard:
						dqn_model.agent_dqn.summary_writer.add_scalar("losses/td_loss", loss, epoch)
						# dqn_model.agent_dqn.summary_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
				
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
				env.reset_init_pos()
				obs, *_ = env.reset()
				done = True
				history += [episode_history]
				episode_rewards = 0
				episode_start = epoch
					
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
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tensorboard', dest='use_tensorboard', action='store_true',
						help='Flag the signals the use of a tensorboard summary writer. Expects argument --tensorboardDetails to be present')
	parser.add_argument('--tensorboardDetails', dest='tensorboard_details', nargs='+', required=False, default=None,
						help='List with the details for the tensorboard summary writer: <log_dirname: str>, <queue_size :int>, <flush_time: int>, <suffix: str>'
							 ' Use only in combination with --tensorboard option')
	
	# Train parameters
	parser.add_argument('--cycles', dest='n_cycles', type=int, default=1,
						help='Number of training cycles, each cycle spawns the field with a different food items configurations.')
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
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--epoch-logging', dest='ep_log', action='store_true', help='')
	parser.add_argument('--train-tags', dest='tags', type=str, nargs='+', required=False, default=None,
						help='List of tags for grouping in weights and biases, empty by default signaling not to train under a specific set of tags')
	
	# Environment parameters
	parser.add_argument('--hunter-ids', dest='hunter_ids', type=str, nargs='+', required=True, help='List with the hunter ids in the environment')
	parser.add_argument('--prey-ids', dest='prey_ids', type=str, nargs='+', required=True, help='List with the prey ids in the environment')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--hunter-classes', dest='hunter_class', type=int, required=True, help='Class of agent to use for the hunters')
	parser.add_argument('--prey-type', dest='prey_type', type=str, required=True, choices=['idle', 'greedy', 'random'],
						help='Type of prey in the environment, possible types: idle, greedy or random')
	parser.add_argument('--n-hunters-catch', dest='require_catch', type=int, required=True, help='Minimum number of hunters required to catch a prey')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag that signals the use of the field render while training')
	parser.add_argument('--catch-reward', dest='catch_reward', type=float, required=False, default=5.0, help='Catch reward for catching a prey')
	
	args = parser.parse_args()
	# DQN args
	n_agents = args.n_agents
	architecture = args.architecture
	buffer_size = args.buffer_size
	gamma = args.gamma
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_vdn = args.use_vdn
	use_cnn = args.use_cnn
	use_tensorboard = args.use_tensorboard
	tensorboard_details = args.tensorboard_details
	
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
	tags = args.tags if args.tags is not None else ''
	
	# Pursuit environment args
	hunter_ids = args.hunter_ids
	prey_ids = args.prey_ids
	field_lengths = args.field_lengths
	max_steps = args.max_steps
	use_render = args.use_render
	require_catch = args.require_catch
	hunter_class = args.hunter_class
	prey_type = args.prey_type
	
	hunters = []
	preys = []
	n_hunters = len(hunter_ids)
	n_preys = len(prey_ids)
	for idx in range(n_hunters):
		hunters += [(hunter_ids[idx], hunter_class)]
	for idx in range(n_preys):
		preys += [(prey_ids[idx], PREY_TYPES[prey_type])]
	
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
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	log_filename = (('train_pursuit_single%s_dqn_%dx%d-field_%d-hunters_%d-preys' %
					 ('_vdn' if use_vdn else '', field_size[0], field_size[1], n_hunters, n_preys)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / 'pursuit_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-hunters' % n_hunters) /
				  ('%d-preys' % n_preys) / now.strftime("%Y%m%d-%H%M%S"))
	
	with open(data_dir / 'performances' / 'pursuit' / ('train_performances%s%s.yaml' % ('_' + prey_type, '_vdn' if use_vdn else '')),
			  mode='r+', encoding='utf-8') as train_file:
		train_performances = yaml.safe_load(train_file)
		field_idx = str(field_size[0]) + 'x' + str(field_size[1])
		hunter_idx = str(n_hunters) + '-hunters'
		prey_idx = str(n_preys) + '-preys'
		train_acc = train_performances[field_idx][hunter_idx][prey_idx]
	
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
	
	#####################
	## Training Models ##
	#####################
	try:
		wandb.init(project='pursuit-optimal', entity='miguel-faria',
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
				   dir=log_dir / 'wandb',
				   name=('%ssingle-l%dx%d-%dh-%dp-%s-' % ('vdn-' if use_vdn else 'independent-', field_size[0], field_size[1], n_hunters, n_preys, prey_type) +
						 now.strftime("%Y%m%d-%H%M%S")),
				   sync_tensorboard=True)
		logger.info('##########################')
		logger.info('Starting Pursuit DQN Train')
		logger.info('##########################')
		n_cycles = n_preys * args.n_cycles
		logger.info('Number of cycles: %d' % n_cycles)
		logger.info('Number of iterations per cycle: %d' % n_iterations)
		logger.info('Environment setup')
		env = TargetPursuitEnv(hunters, preys, field_size, sight, prey_ids[0], require_catch, max_steps, use_layer_obs=True, agent_centered=True,
							   catch_reward=args.catch_reward)
		# env = PursuitEnv(hunters, preys, field_size, sight, require_catch, max_steps, use_layer_obs=True)
		logger.info('Setup multi-agent DQN')
		if isinstance(env.observation_space, gymnasium.spaces.MultiBinary):
			obs_space = gymnasium.spaces.MultiBinary([*env.observation_space.shape[1:]])
		else:
			obs_space = env.observation_space[0]
		action_dim = env.action_space[0].n
		agent_action_space = gymnasium.spaces.MultiDiscrete([action_dim] * env.n_hunters)
		if use_vdn:
			dqn_model = SingleModelMADQN(n_agents, action_dim, n_layers, nn.relu, layer_sizes, buffer_size, gamma, agent_action_space,
										 env.observation_space, use_gpu, dueling_dqn, use_ddqn, use_vdn, use_cnn, False, use_tensorboard,
										 tensorboard_details + ['%dh-%dp-%dc' % (n_hunters, n_preys, require_catch)], cnn_properties=cnn_properties)
		else:
			dqn_model = SingleModelMADQN(n_agents, action_dim, n_layers, nn.relu, layer_sizes, buffer_size, gamma, agent_action_space, obs_space,
										 use_gpu, dueling_dqn, use_ddqn, use_vdn, use_cnn, False, use_tensorboard,
										 tensorboard_details + ['%dh-%dp-%dc' % (n_hunters, n_preys, require_catch)], cnn_properties=cnn_properties)
		random.seed(RNG_SEED)
		preys_list = [random.choice(prey_ids) for _ in range(n_cycles)]
		logger.info('Starting training')
		for cycle in range(n_cycles):
			logger.info('Cycle %d of %d' % (cycle+1, n_cycles))
			env.seed(RNG_SEED)
			sys.stdout.flush()
			if cycle == 0:
				cycle_init_eps = initial_eps
			else:
				cycle_init_eps = eps_cycle_schedule('exp', cycle, n_cycles, initial_eps, final_eps, cycle_decay)
			env.target = preys_list[cycle]
			logger.info('Starting exploration: %d with decay of %f' % (cycle_init_eps, eps_decay))
			cycle_warmup = warmup * 0.5 ** min(cycle, 1)
			cnn_shape = (0,) if not dqn_model.agent_dqn.cnn_layer else (*obs_space.shape[1:], obs_space.shape[0])
			history = train_pursuit_dqn(dqn_model, env, n_iterations, max_steps * n_iterations, batch_size, learn_rate, target_update_rate, cycle_init_eps,
										final_eps, eps_type, RNG_SEED, logger, cnn_shape, eps_decay, cycle_warmup, train_freq, target_freq, tensorboard_freq,
										use_render, cycle)
			
		logger.info('Saving final model')
		dqn_model.save_model(('preys_%d' % n_preys), model_path, logger)
		sys.stdout.flush()
	
		####################
		## Testing Model ##
		####################
		env = TargetPursuitEnv(hunters, preys, field_size, sight, prey_ids[0], require_catch, max_steps, use_layer_obs=True)
		env.seed(TEST_RNG_SEED)
		np.random.seed(TEST_RNG_SEED)
		random.seed(TEST_RNG_SEED)
		# failed_history = []
		tests_passed = 0
		testing_prey_lists = [random.choice(prey_ids) for _ in range(N_TESTS)]
		for n_test in range(N_TESTS):
			env.reset_init_pos()
			obs, *_ = env.reset()
			logger.info('Test number %d' % (n_test + 1))
			logger.info('Prey locations: ' + ', '.join(['(%d, %d)' % env.agents[prey_id].pos for prey_id in env.prey_alive_ids]))
			logger.info('Agent positions: ' + ', '.join(['(%d, %d)' % env.agents[hunter_id].pos for hunter_id in env.hunter_ids]))
			logger.info('Testing sequence: ' + ', '.join(testing_prey_lists[n_test]))
			obs, *_ = env.reset()
			epoch = 0
			agent_reward = [0] * n_hunters
			game_over = False
			finished = False
			timeout = False
			env.target = testing_prey_lists[n_test]
			cnn_shape = (0,) if not dqn_model.agent_dqn.cnn_layer else (*obs_space.shape[1:], obs_space.shape[0])
			while not game_over:
				
				actions = []
				for h_idx in range(n_hunters):
					if dqn_model.agent_dqn.cnn_layer:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params,
																	   obs[h_idx].reshape((1, *cnn_shape)))[0]
					else:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, obs[h_idx])
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
					actions += [action]
				for prey_id in env.prey_alive_ids:
					actions += [env.agents[prey_id].act(env)]
				actions = np.array(actions)
				next_obs, rewards, finished, timeout, infos = env.step(actions)
				for i in range(n_hunters):
					agent_reward[i] = rewards[i]
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
				logger.info('Test %d timed out' % (n_test + 1))
		
		env.close()
		logger.info('Passed %d tests out of %d' % (tests_passed, N_TESTS))
		
		if (tests_passed / N_TESTS) > train_acc:
			logger.info('Updating best model for current loc')
			Path.mkdir(model_path.parent.absolute() / 'best', parents=True, exist_ok=True)
			dqn_model.save_model('all_preys', model_path.parent.absolute() / 'best', logger)
			train_acc = tests_passed / N_TESTS
		
		gc.collect()
		wandb.finish()
		with open(data_dir / 'performances' / 'pursuit' / ('train_performances%s%s.yaml' % ('_' + prey_type, '_vdn' if use_vdn else '')),
				  mode='r+', encoding='utf-8') as train_file:
			performance_data = yaml.safe_load(train_file)
			field_idx = str(field_size[0]) + 'x' + str(field_size[1])
			hunter_idx = str(n_hunters) + '-hunters'
			prey_idx = str(n_preys) + '-preys'
			performance_data[field_idx][hunter_idx][prey_idx] = train_acc
			train_file.seek(0)
			sorted_data = dict(
				[[sorted_key, performance_data[sorted_key]] for sorted_key in
				 [str(t[0]) + 'x' + str(t[1]) for t in sorted([tuple([int(x) for x in key.split('x')]) for key in performance_data.keys()])]])
			yaml.safe_dump(sorted_data, train_file)
		
	except KeyboardInterrupt as ks:
		logger.info('Caught keyboard interrupt, cleaning up and closing.')
		wandb.finish()
		with open(data_dir / 'performances' / 'pursuit' / ('train_performances%s%s.yaml' % ('_' + prey_type, '_vdn' if use_vdn else '')),
				  mode='r+', encoding='utf-8') as train_file:
			performance_data = yaml.safe_load(train_file)
			field_idx = str(field_size[0]) + 'x' + str(field_size[1])
			hunter_idx = str(n_hunters) + '-hunters'
			prey_idx = str(n_preys) + '-preys'
			performance_data[field_idx][hunter_idx][prey_idx] = train_acc
			train_file.seek(0)
			sorted_data = dict(
				[[sorted_key, performance_data[sorted_key]] for sorted_key in
				 [str(t[0]) + 'x' + str(t[1]) for t in sorted([tuple([int(x) for x in key.split('x')]) for key in performance_data.keys()])]])
			yaml.safe_dump(sorted_data, train_file)


if __name__ == '__main__':
	main()

