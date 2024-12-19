#! /usr/bin/env python
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
import wandb

from dl_algos.single_model_madqn import LegibleSingleMADQN
from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_envs.pursuit.pursuit_env import TargetPursuitEnv, Action
from dl_envs.pursuit.agents.agent import Agent, AgentType
from pathlib import Path
from itertools import permutations
from typing import List, Tuple, Optional
from datetime import datetime
from gymnasium.spaces import MultiBinary, MultiDiscrete
from wandb.wandb_run import Run

RNG_SEED = 6102023
TEST_RNG_SEED = 4072023
N_TESTS = 100
PREY_TYPES = {'idle': 0, 'greedy': 1, 'random': 2}
MIN_TRAIN_PERFORMANCE = 0.9


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


def train_pursuit_legible_dqn(dqn_model: LegibleSingleMADQN, env: TargetPursuitEnv, num_iterations: int, batch_size: int, optim_learn_rate: float,
							  tau: float, initial_eps: float, final_eps: float, eps_type: str, reward_type: str, rng_seed: int, logger: logging.Logger, cnn_shape: Tuple[int],
							  exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1,
							  use_render: bool = False, greedy_action: bool = True, sofmax_temp: float = 1.0, initial_model_path: str = '', use_tracker: bool = False,
							  performance_tracker: Optional[Run] = None, tracker_panel: str = '', debug: bool = False, dqn_target_key: str = '') -> None:

	rng_gen = np.random.default_rng(rng_seed)
	
	# Setup DQNs for training
	obs, *_ = env.reset()
	if not dqn_model.agent_dqn.dqn_initialized:
		if dqn_model.agent_dqn.cnn_layer:
			dqn_model.agent_dqn.init_network_states(rng_seed, obs[0].reshape((1, *cnn_shape)), optim_learn_rate, initial_model_path)
		else:
			dqn_model.agent_dqn.init_network_states(rng_seed, obs[0], optim_learn_rate, initial_model_path)

	start_time = time.time()
	sys.stdout.flush()
	epoch = 0
	start_record_it = 0
	start_record_epoch = 0
	avg_episode_len = []

	try:
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			episode_start = epoch
			avg_loss = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			logger.info('Hunters: ' + ', '.join(['%s @ (%d, %d)' % (hunter_id, *env.agents[hunter_id].pos) for hunter_id in env.hunter_ids]))
			logger.info('Number of preys spawn:\t%d' % env.n_preys_alive)
			logger.info('Starting prey locations: ' + ', '.join(['%s @ (%d, %d)' % (prey_id, *env.agents[prey_id].pos) for prey_id in env.prey_alive_ids]))
			logger.info('Objective prey: %s @ (%d, %d)' % (env.target, *env.agents[env.target].pos))
			eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			dqn_target = env.target
			while not done:
	
				# interact with environment
				explore = rng_gen.random() < eps
				if explore:
					actions = np.array(list(env.action_space.sample()[:env.n_hunters]) + [env.agents[prey_id].act(env) for prey_id in env.prey_alive_ids])
				else:
					actions = []
					for a_idx in range(env.n_hunters):
	
						if a_idx < dqn_model.n_leg_agents:
							online_params = dqn_model.agent_dqn.online_state.params
						else:
							online_params = dqn_model.optimal_models[dqn_target_key].params
	
						if dqn_model.agent_dqn.cnn_layer:
							cnn_obs = obs[a_idx].reshape((1, *cnn_shape))
							q_values = dqn_model.agent_dqn.q_network.apply(online_params, cnn_obs)[0]
						else:
							q_values = dqn_model.agent_dqn.q_network.apply(online_params, obs[a_idx])
						
						if greedy_action:
							action = q_values.argmax(axis=-1)
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						# action = jax.device_get(action)
						episode_q_vals += (float(q_values[int(action)]) / env.n_hunters)
						actions += [action]
					
					for prey_id in env.prey_alive_ids:
						actions += [env.agents[prey_id].act(env)]
	
					actions = np.array(actions)
				
				if debug:
					logger.info(env.get_env_log() + 'Actions: ' + str([Action(act).name for act in actions]) + ' Explored? %r' % explore + '\n')
				
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				
				if debug:
					logger.info(env.get_env_log() + 'Terminated? %r ' % terminated + 'Timedout? %r' % timeout + '\n')
				
				if use_render:
					env.render()
				
				legible_rewards = np.zeros(dqn_model.num_agents)
				live_goals = env.prey_ids
				n_goals = len(env.prey_ids)
				if n_goals > 1:
					for a_idx in range(dqn_model.num_agents):
						act_q_vals = np.zeros(n_goals)
						action = actions[a_idx]
						goal_action_q = 0.0
						for g_idx in range(n_goals):
							if dqn_target == live_goals[g_idx]:
								if dqn_model.agent_dqn.cnn_layer:
									obs_reshape = obs[a_idx].reshape((1, *cnn_shape))
									q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_target_key].params, obs_reshape)[0]
								else:
									q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_target_key].params, obs[a_idx])
								goal_action_q = q_vals[action]
							else:
								goal_obs = env.make_target_grid_obs(live_goals[g_idx])
								if dqn_model.agent_dqn.cnn_layer:
									obs_reshape = goal_obs[a_idx].reshape((1, *cnn_shape))
									q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_target_key].params, obs_reshape)[0]
								else:
									q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[dqn_target_key].params, goal_obs[a_idx])
							act_q_vals[g_idx] = np.exp(dqn_model.beta * (q_vals[action] - q_vals.max()) / sofmax_temp)
						if reward_type == 'reward':
							legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_target)] / act_q_vals.sum()) * rewards[a_idx]
						elif reward_type == 'q_vals':
							legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_target)] / act_q_vals.sum()) * goal_action_q
						elif reward_type == 'info':
							legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_target)] / act_q_vals.sum()) + rewards[a_idx]
						else:
							legible_rewards[a_idx] = act_q_vals[live_goals.index(dqn_target)] / act_q_vals.sum()
				else:
					for a_idx in range(dqn_model.n_leg_agents):
						legible_rewards[a_idx] = rewards[a_idx]
				
				if terminated or ('caught_target' in infos.keys() and infos['caught_target'] == env.target):
					finished = np.ones(dqn_model.num_agents)
					legible_rewards = legible_rewards / (1 - dqn_model.agent_dqn.gamma)
				else:
					finished = np.zeros(dqn_model.num_agents)
				
				# logger.info(str(finished) + '\tTimestep: %d' % env.env_timestep)
				
				# store new samples
				if dqn_model.use_vdn:
					hunter_actions = actions[:env.n_hunters]
					store_rewards = np.hstack((legible_rewards[:dqn_model.n_leg_agents], rewards[dqn_model.n_leg_agents:env.n_hunters]))
					dqn_model.replay_buffer.add(obs, next_obs, hunter_actions, store_rewards, finished[0], [])
				else:
					for a_idx in range(env.n_hunters):
						dqn_model.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], legible_rewards[a_idx], finished[a_idx], [])
				episode_rewards += sum(legible_rewards[:dqn_model.n_leg_agents]) / dqn_model.n_leg_agents
	
				if use_tracker:
					performance_tracker.log({
							tracker_panel + "-charts/performance/legible_reward": 	sum(legible_rewards) / dqn_model.n_leg_agents,
							tracker_panel + "-charts/performance/reward": 			sum(rewards[:env.n_hunters]) / env.n_hunters},
							step=(epoch + episode_start))
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
					episode_len = epoch - episode_start
					avg_episode_len += [episode_len]
					if use_tracker:
						performance_tracker.log({
								tracker_panel + "-charts/performance/mean_episode_q_vals": episode_q_vals / episode_len,
								tracker_panel + "-charts/performance/mean_episode_return": episode_rewards / episode_len,
								tracker_panel + "-charts/performance/episodic_length":     episode_len,
								tracker_panel + "-charts/performance/avg_episode_length":  np.mean(avg_episode_len),
								tracker_panel + "-charts/control/iteration":               it,
								tracker_panel + "-charts/control/exploration":             eps,
								tracker_panel + "-charts/losses/td_loss":                  sum(avg_loss) / max(len(avg_loss), 1)
						},
								step=(it + start_record_it))
					logger.info("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (epoch - episode_start, eps, episode_rewards))
					env.target = rng_gen.choice(env.prey_ids)
					obs, *_ = env.reset()
					done = True
					episode_rewards = 0
					episode_start = epoch
					epoch = 0
		
	except ValueError as err:
		logger.error(err)
		return
	

# noinspection DuplicatedCode
def main():
	parser = argparse.ArgumentParser(description='Train DQN for LB Foraging with fixed foods in environment')
	
	# Multi-agent DQN params
	parser.add_argument('--n-agents', dest='n_agents', type=int, required=True, help='Number of agents in the environment')
	parser.add_argument('--n-leg-agents', dest='n_leg_agents', type=int, default=1, help='Number of legible agents in the environment')
	parser.add_argument('--architecture', dest='architecture', type=str, required=True, help='DQN architecture to use from the architectures yaml')
	parser.add_argument('--buffer', dest='buffer_size', type=int, required=True, help='Size of the replay buffer in the DQN')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--beta', dest='beta', type=float, required=False, default=0.9, help='Constant that defines how close follow optimal')
	parser.add_argument('--gpu', dest='use_gpu', action='store_true', help='Flag that signals the use of gpu for the training')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--tracker', dest='use_tracker', action='store_true', help='Flag the signals the use of the wandb performance tracker.')
	parser.add_argument('--reward', dest='leg_reward', type=str, required=True, choices=['simple', 'reward', 'q_vals', 'info'],
	                    help='Type of legible reward signal to use from amongst:\n\t\'simple\' - use just the objective saliency as reward'
	                         '\n\t\'reward\' - weight environment reward with objective saliency\n\t\'q_vals\' - weight agent q_vals with objective saliency'
	                         '\n\t\'info\' - information based legibility by summing environment reward with objective saliency')
	
	# Train parameters
	parser.add_argument('--train-performance', dest='min_train_performance', type=float, default=MIN_TRAIN_PERFORMANCE,
	                    help='Minimum performance threshold to skip model train')
	parser.add_argument('--iterations', dest='n_iterations', type=int, required=True, help='Number of iterations to run training')
	parser.add_argument('--batch', dest='batch_size', type=int, required=True, help='Number of samples in each training batch')
	parser.add_argument('--train-freq', dest='train_freq', type=int, required=True, help='Number of epochs between each training update')
	parser.add_argument('--target-freq', dest='target_freq', type=int, required=True, help='Number of epochs between updates to target network')
	parser.add_argument('--alpha', dest='learn_rate', type=float, required=False, default=2.5e-4, help='Learn rate for DQN\'s Q network')
	parser.add_argument('--tau', dest='target_learn_rate', type=float, required=False, default=2.5e-6, help='Learn rate for the target network')
	parser.add_argument('--init-eps', dest='initial_eps', type=float, required=False, default=1., help='Exploration rate when training starts')
	parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=0.05, help='Minimum exploration rate for training')
	parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=0.5, help='Decay rate for the exploration update')
	parser.add_argument('--legibility-temp', dest='temp', type=float, required=False, default=0.5, help='Temperature parameter for legibility softmax')
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
	parser.add_argument('--use-opt-vdn', dest='opt_vdn', action='store_true', help='Signal the use of optimal models with a VDN architecture')
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--train-targets', dest='train_targets', type=str, nargs='+', required=False, default=None,
						help='List with the prey ids to train to catch')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
						help='Directory to store trained models and load optimal models, if left blank stored in default location')
	parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
	                    help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
	parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
	parser.add_argument('--tracker-dir', dest='tracker_dir', type=str, default='', help='Path to the directory to store the tracker data')
	parser.add_argument('--use-lower-model', dest='use_lower_model', action='store_true',
	                    help='Flag that signals using curriculum learning using a model with one less food item spawned (when using with only 1 item, defaults to false).')
	parser.add_argument('--use-higher-model', dest='use_higher_model', action='store_true',
	                    help='Flag that signals using curriculum learning using a model with one more food item spawned (when using with only all items, defaults to false).')
	parser.add_argument('--buffer-smart-add', dest='buffer_smart_add', action='store_true',
	                    help='Flag denoting the use of smart sample add to experience replay buffer instead of first-in first-out')
	parser.add_argument('--buffer-method', dest='buffer_method', type=str, required=False, default='uniform', choices=['uniform', 'weighted'],
	                    help='Method of deciding how to add new experience samples when replay buffer is full')
	
	# Environment parameters
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--hunter-classes', dest='hunter_class', type=int, required=True, help='Class of agent to use for the hunters')
	parser.add_argument('--hunter-ids', dest='hunter_ids', type=str, nargs='+', required=True, help='List with the hunter ids in the environment')
	parser.add_argument('--n-hunters-catch', dest='require_catch', type=int, required=True, help='Minimum number of hunters required to catch a prey')
	parser.add_argument('--prey-ids', dest='prey_ids', type=str, nargs='+', required=True, help='List with the prey ids in the environment')
	parser.add_argument('--prey-type', dest='prey_type', type=str, required=True, choices=['idle', 'greedy', 'random'],
						help='Type of prey in the environment, possible types: idle, greedy or random')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag that signals the use of the field render while training')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--catch-reward', dest='catch_reward', type=float, required=False, default=5.0, help='Catch reward for catching a prey')
	args = parser.parse_args()

	# DQN args
	n_agents = args.n_agents
	n_leg_agents = args.n_leg_agents
	architecture = args.architecture
	buffer_size = args.buffer_size
	gamma = args.gamma
	beta = args.beta
	use_gpu = args.use_gpu
	dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_cnn = args.use_cnn
	use_vdn = args.use_vdn
	use_tracker = args.use_tracker
	tracker_dir = args.tracker_dir
	leg_reward = args.leg_reward
	
	# Train args
	train_thresh = args.min_train_performance
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
	restart_train = args.restart_train
	restart_info = args.restart_info
	debug = args.debug
	temp = args.temp
	optim_vdn = args.opt_vdn
	use_lower_model = args.use_lower_model
	use_higher_model = args.use_higher_model
	
	# LB-Foraging environment args
	hunter_ids = args.hunter_ids
	prey_ids = args.prey_ids
	field_lengths = args.field_lengths
	max_steps = args.max_steps
	use_render = args.use_render
	require_catch = args.require_catch
	hunter_class = args.hunter_class
	prey_type = args.prey_type

	try:
		assert not (use_higher_model and use_lower_model)

	except AssertionError:
		print('Attempt at using curriculum learning using both model trained with one more and one less food item spawned')
		return

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
	home_dir = Path(__file__).parent.absolute().parent.absolute()
	log_dir = Path(args.logs_dir) if args.logs_dir != '' else home_dir / 'logs'
	data_dir = Path(args.data_dir) if args.data_dir != '' else home_dir / 'data'
	models_dir = Path(args.models_dir) if args.models_dir != '' else home_dir / 'models'
	if tracker_dir == '':
		tracker_dir = log_dir
	log_filename = (('train_pursuit_legible%s_dqn_%dx%d-field_%d-hunters_%d-preys' % ('_vdn' if use_vdn else '', field_size[0], field_size[1], n_hunters, n_preys)) +
					'_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / ('pursuit_legible%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
	              ('%d-hunters' % n_hunters) / now.strftime("%Y%m%d-%H%M%S"))
	optim_dir = (models_dir / ('pursuit_single%s_dqn' % ('_vdn' if optim_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
	             ('%d-hunters' % n_hunters) / ('%s' % prey_type) / 'best')

	with open(data_dir / 'performances' / 'pursuit' / ('train_legible_performances%s%s.yaml' % ('_' + prey_type, '_vdn' if use_vdn else '')),
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

	logger.info('##################################')
	logger.info('Starting Pursuit Legible DQN Train')
	logger.info('##################################')
	optim_models = {}
	model_names = [fname.name for fname in optim_dir.iterdir()]
	target_name = 'prey'
	model_found = False
	for name in model_names:
		if name.find("%d-preys" % n_preys) != -1:
			goal = name
			model_found = True
			break

	if model_found:
		optim_models[target_name] = goal
	else:
		logger.error('Optimal model for %d %s preys not found. Exiting script' % (n_preys, prey_type))
		return
	
	#####################
	## Training Models ##
	#####################
	if train_acc <= train_thresh:
		try:
			wandb_run = wandb.init(project='pursuit-legible',
								   entity='miguel-faria',
								   config={
									   "field": "%dx%d" % (field_size[0], field_size[1]),
									   "preys": n_preys,
									   "hunters": n_hunters,
									   "legible_agents": n_leg_agents,
									   "online_learing_rate": learn_rate,
									   "target_learning_rate": target_update_rate,
									   "discount": gamma,
									   "eps_decay_type": eps_type,
									   "eps_decay": eps_decay,
									   "dqn_architecture": architecture,
									   "iterations": n_iterations,
									   "beta": beta,
									   "softmax_temp": temp,
									   "buffer_size": buffer_size,
									   "buffer_add": "smart" if args.buffer_smart_add else "plain",
									   "buffer_add_method": args.buffer_method if args.buffer_smart_add else "standard",
									   "reward_type": leg_reward,
									   "batch_size": batch_size,
									   "curriculum_learning": 'no' if not (use_higher_model or use_higher_model) else ('lower_model' if use_lower_model else 'higher_model')
								   },
								   dir=tracker_dir,
								   name=('%ssingle-l%dx%d-%dh-%dp-%s-' % ('vdn-' if use_vdn else 'independent-', field_size[0], field_size[1], n_hunters, n_preys, prey_type) +
										 now.strftime("%Y%m%d-%H%M%S")),
								   sync_tensorboard=True,
								   settings=wandb.Settings(init_timeout=120))
			logger.info('##########################')
			logger.info('Starting Pursuit DQN Train')
			logger.info('##########################')
			logger.info('Number of iterations per cycle: %d' % n_iterations)
			logger.info('Environment setup')
			env = TargetPursuitEnv(hunters, preys, field_size, sight, prey_ids[0], require_catch, max_steps, use_layer_obs=True, agent_centered=True,
								   catch_reward=args.catch_reward)
			logger.info('Setup multi-agent DQN')
			agent_action_space = env.action_space[0]
			n_actions = agent_action_space.n
			if isinstance(env.observation_space, MultiBinary):
				obs_space = MultiBinary([*env.observation_space.shape[1:]])
			else:
				obs_space = env.observation_space[0]
			if use_vdn:
				action_space = MultiDiscrete([agent_action_space.n] * env.n_hunters)
				agent_madqn = LegibleSingleMADQN(n_hunters, n_actions, n_layers, nn.relu, layer_sizes, buffer_size, gamma, beta, action_space, env.observation_space, use_gpu,
				                                 False, optim_dir, optim_models, target_name, dueling_dqn, use_ddqn, use_vdn, use_cnn,
				                                 n_legible_agents=min(n_leg_agents, n_hunters), cnn_properties=cnn_properties,
				                                 buffer_data=(args.buffer_smart_add, args.buffer_method))
			else:
				agent_madqn = LegibleSingleMADQN(n_hunters, n_actions, n_layers, nn.relu, layer_sizes, buffer_size, gamma, beta, agent_action_space, obs_space, use_gpu,
				                                 False, optim_dir, optim_models, target_name, dueling_dqn, use_ddqn, use_vdn, use_cnn,
				                                 n_legible_agents=min(n_leg_agents, n_hunters), cnn_properties=cnn_properties,
				                                 buffer_data=(args.buffer_smart_add, args.buffer_method))

			if use_lower_model and n_preys > 1:
				prev_model_path = model_path.parent.absolute() / 'best'
				logger.info('Model pahth: ' + str(prev_model_path))
				if (prev_model_path / ('%d-preys_single_model.model' % max(n_preys - 1, 1))).exists():
					logger.info('Using model trained with %d foods spawned as a baseline' % max(n_preys - 1, 1))
					curriculum_model_path = str(prev_model_path / ('%d-preys_single_model.model' % max(n_preys - 1, 1)))
				else:
					logger.info('Model with one less prey not found, training from scratch')
					curriculum_model_path = ''
			elif use_higher_model and n_preys < n_preys:
				next_model_path = model_path.parent.absolute() / 'best'
				if (next_model_path / ('%d-preys_single_model.model' % min(n_preys + 1, n_preys))).exists():
					logger.info('Using model trained with %d foods spawned as a baseline' % min(n_preys + 1, n_preys))
					curriculum_model_path = str(next_model_path / ('%d-preys_single_model.model' % min(n_preys + 1, n_preys)))
				else:
					logger.info('Model with one more prey not found, training from scratch')
					curriculum_model_path = ''
			else:
				logger.info('Training model from scratch')
				curriculum_model_path = ''

			random.seed(RNG_SEED)
			logger.info('Starting training')
			env.seed(RNG_SEED)
			sys.stdout.flush()
			cnn_shape = (0,) if not agent_madqn.agent_dqn.cnn_layer else (*obs_space.shape[1:], obs_space.shape[0])
			tracker_panel = 'l%dx%d-%dp' % (field_size[0], field_size[1], n_preys)
			greedy_actions = False
			train_pursuit_legible_dqn(agent_madqn, env, n_iterations, batch_size, learn_rate, target_update_rate, initial_eps, final_eps, eps_type,
			                          leg_reward, RNG_SEED, logger, cnn_shape, eps_decay, warmup, train_freq, target_freq, tensorboard_freq, False,
			                          greedy_actions, temp, curriculum_model_path, use_tracker, wandb_run, tracker_panel, debug, target_name)

			logger.info('Saving final model')
			agent_madqn.save_model(('preys-%d' % n_preys), model_path, logger)
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
			testing_prey_lists = [random.choice(prey_ids) for _ in range(N_TESTS)]
			for n_test in range(N_TESTS):
				env.reset_init_pos()
				env.target = testing_prey_lists[n_test]
				obs, *_ = env.reset()
				logger.info('Test number %d' % (n_test + 1))
				logger.info('Prey locations: ' + ', '.join(['(%d, %d)' % env.agents[prey_id].pos for prey_id in env.prey_alive_ids]))
				logger.info('Agent positions: ' + ', '.join(['(%d, %d)' % env.agents[hunter_id].pos for hunter_id in env.hunter_ids]))
				logger.info('Testing sequence: ' + ', '.join(testing_prey_lists[n_test]))
				# obs, *_ = env.reset()
				epoch = 0
				agent_reward = [0] * n_hunters
				game_over = False
				finished = False
				timeout = False
				cnn_shape = (0,) if not agent_madqn.agent_dqn.cnn_layer else (*obs_space.shape[1:], obs_space.shape[0])
				while not game_over:

					actions = []
					for h_idx in range(n_hunters):
						dqn = agent_madqn.agent_dqn
						if h_idx < agent_madqn.n_leg_agents:
							online_params = agent_madqn.agent_dqn.online_state.params
						else:
							online_params = agent_madqn.optimal_models[agent_madqn.goal].params

						if dqn.cnn_layer:
							obs_reshape = obs[h_idx].reshape((1, *cnn_shape))
							q_values = dqn.q_network.apply(online_params, obs_reshape)[0]
						else:
							q_values = dqn.q_network.apply(online_params, obs[h_idx])
						action = q_values.argmax(axis=-1)
						action = jax.device_get(action)
						actions += [action]
					for prey_id in env.prey_alive_ids:
						actions += [env.agents[prey_id].act(env)]
					actions = np.array(actions)
					next_obs, rewards, finished, timeout, infos = env.step(actions)
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

			logger.info('Passed %d tests out of %d' % (tests_passed, N_TESTS))
			logger.info('Failed tests history:')
			logger.info(failed_history)
			
			if (tests_passed / N_TESTS) > train_acc:
				logger.info('Updating best model for current loc')
				Path.mkdir(model_path.parent.absolute() / 'best', parents=True, exist_ok=True)
				agent_madqn.save_model('%d-preys' % n_preys, model_path.parent.absolute() / 'best', logger)
				train_acc = tests_passed / N_TESTS

			wandb_run.finish()
			with open(data_dir / 'performances' / 'pursuit' / ('train_legible_performances_%s%s.yaml' % ('_' + prey_type, '_vdn' if use_vdn else '')),
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
			with open(data_dir / 'performances' / 'pursuit' / ('train_legible_performances_%s%s.yaml' % ('_' + prey_type, '_vdn' if use_vdn else '')),
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
				
		finally:
			wandb.finish()


if __name__ == '__main__':
	main()

