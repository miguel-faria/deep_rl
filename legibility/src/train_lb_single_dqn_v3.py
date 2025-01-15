#! /usr/bin/env python

import os
import sys
import argparse
import numpy as np
import flax.linen as nn
import yaml
import jax
import math
import logging
import time
import gc
import wandb

from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_algos.single_model_madqn import SingleModelMADQN
from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import Action
from pathlib import Path
from itertools import product
from typing import List, Tuple, Optional
from datetime import datetime
from gymnasium.spaces import MultiBinary, MultiDiscrete
from wandb.wandb_run import Run


RNG_SEED = 13042023
TEST_RNG_SEED = 4072023
N_TESTS = 100
MIN_TRAIN_PERFORMANCE = 0.9


def input_callback(env: FoodCOOPLBForaging, stop_flag: bool):
	try:
		while not stop_flag:
			command = input('Interactive commands:\n\trender - display renderization of the interaction\n\tstop_render - stops the renderization\nCommand: ')
			if command == 'render':
				env.use_render = True
			elif command == 'stop_render':
				if env.use_render:
					env.use_render = False
	
	except KeyboardInterrupt as ki:
		return


def number_food_combinations(max_foods: int, n_foods_spawn: int) -> int:
	return max(1, int(math.factorial(max_foods) / (math.factorial(n_foods_spawn) * math.factorial(max_foods - n_foods_spawn)) * 0.75))


def eps_cycle_schedule(cycle_nr: int, max_cycles: int, init_eps: float, final_eps: float, decay_rate: float, sched_type: str = 'log') -> float:
	if sched_type == 'linear':
		return max(((final_eps - init_eps) / max_cycles) * cycle_nr / decay_rate + init_eps, final_eps)
	else:
		return max(init_eps - decay_rate ** ((max_cycles - 1) / cycle_nr), final_eps)


def cycle_foods_spawn(max_foods_spawn: int, cycle: int, rng_gen: np.random.Generator) -> int:
	if cycle < 2:
		return max_foods_spawn
	else:
		return int(rng_gen.integers(1, max_foods_spawn + 1))


def get_history_entry(obs: np.ndarray, actions: List[int], n_agents: int) -> List:
	entry = []
	for a_idx in range(n_agents):
		state_str = ' '.join([str(int(x)) for x in obs[a_idx]])
		action = actions[a_idx]
		entry += [state_str, str(action)]
	
	return entry


def train_lb_model(env: FoodCOOPLBForaging, dqn_model: SingleModelMADQN, num_iterations: int, max_timesteps: int, n_foods_spawn: int, batch_size: int, online_lr: float,
                   target_lr: float, initial_eps: float, final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, cnn_shape: Tuple[int],
                   exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100, tracker_frequency: int = 1, use_render: bool = False,
                   greedy_action: bool = True, epoch_logging: bool = False, initial_model_path: str = '', use_tracker: bool = False, performance_tracker: Optional[Run] = None,
                   tracker_panel: str = '', debug: bool = False) -> None:
	
	rng_gen = np.random.default_rng(rng_seed)
	
	# Setup DQNs for training
	# env.set_objective(env.foods[rng_gen.integers(n_foods_spawn)].position)
	env.spawn_players([env.max_player_level] * env.n_players)
	env.spawn_food(n_foods_spawn, env.max_food_level)
	obs, *_ = env.reset()
	if not dqn_model.agent_dqn.dqn_initialized:
		dqn_model.initialize_network(cnn_shape, logger, obs, online_lr, rng_seed, initial_model_path)
	
	start_time = time.time()
	epoch = 0
	sys.stdout.flush()
	start_record_it = 0
	start_record_epoch = 0
	avg_episode_len = []
	
	for it in range(num_iterations):
		if use_render:
			env.render()
		done = False
		episode_rewards = 0
		episode_q_vals = 0
		episode_start = epoch
		avg_loss = []
		logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
		logger.info('Agents: ' + ', '.join(['%s @ (%d, %d) with level %d' % (player.player_id, *player.position, player.level) for player in env.players]))
		logger.info('Number of food spawn:\t%d' % n_foods_spawn)
		logger.info('Food items: ' + ', '.join(['(%d, %d) with level: %d' % (*food.position, food.level) for food in env.foods]))
		logger.info('Food objective: (%d, %d)' % env.obj_food)
		while not done:
			
			# interact with environment
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			
			explore = rng_gen.random() < eps
			if explore:  # Exploration
				actions = np.array(env.action_space.sample())
			else:  # Exploitation
				actions = []
				for a_idx in range(env.n_players):
					if dqn_model.agent_dqn.cnn_layer:
						cnn_obs = obs[a_idx].reshape((1, *cnn_shape))
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, cnn_obs)[0]
					else:
						q_values = dqn_model.agent_dqn.q_network.apply(dqn_model.agent_dqn.online_state.params, obs[a_idx])
					
					if greedy_action:
						action = q_values.argmax()
					else:
						pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
						pol = pol / pol.sum()
						action = rng_gen.choice(range(env.action_space[0].n), p=pol)
					
					episode_q_vals += (float(q_values[int(action)]) / dqn_model.num_agents)
					actions += [action]
				actions = np.array(actions)
			
			if debug:
				logger.info(env.get_env_log() + 'Actions: ' + str([Action(act).name for act in actions]) + ' Explored? %r' % explore + '\n')
			
			next_obs, rewards, terminated, timeout, infos = env.step(actions)
			if use_render:
				env.render()
			
			if len(rewards) == 1:
				rewards = np.array([rewards] * dqn_model.num_agents)
			
			if terminated:
				finished = np.ones(dqn_model.num_agents)
			else:
				finished = np.zeros(dqn_model.num_agents)
			
			# store new samples
			if dqn_model.use_vdn:
				dqn_model.replay_buffer.add(obs, next_obs, actions, rewards, finished[0], infos)
				episode_rewards += sum(rewards) / dqn_model.num_agents
			else:
				for a_idx in range(dqn_model.num_agents):
					dqn_model.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], rewards[a_idx], finished[a_idx], infos)
					episode_rewards += (rewards[a_idx] / dqn_model.num_agents)
			if use_tracker and epoch_logging:
				performance_tracker.log({tracker_panel + "-charts/performance/reward": sum(rewards)}, step=(epoch + start_record_epoch))
			obs = next_obs
			
			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					loss = jax.device_get(dqn_model.update_model(batch_size, epoch, start_time, tracker_frequency, logger, cnn_shape))
					if use_tracker and epoch_logging:
						performance_tracker.log({tracker_panel + "-charts/losses/td_loss": loss}, step=epoch)
					else:
						avg_loss += [loss]
				
				if epoch % target_freq == 0:
					dqn_model.agent_dqn.update_target_model(target_lr)
			
			epoch += 1
			sys.stdout.flush()
			
			# Check if iteration is over
			if terminated or timeout:
				episode_len = epoch - episode_start
				avg_episode_len += [episode_len]
				if use_tracker:
					performance_tracker.log(
							data={
									tracker_panel + "-charts/performance/mean_episode_q_vals": episode_q_vals / episode_len,
									tracker_panel + "-charts/performance/mean_episode_return": episode_rewards / episode_len,
									tracker_panel + "-charts/performance/episodic_length":     episode_len,
									tracker_panel + "-charts/performance/avg_episode_length":  np.mean(avg_episode_len),
							},
							step=(it + start_record_it))
					performance_tracker.log(
							data={
									tracker_panel + "-charts/control/iteration":               it,
									tracker_panel + "-charts/control/exploration":             eps,
							},
							step=(it + start_record_it))
					if not epoch_logging:
						performance_tracker.log({tracker_panel + "-charts/losses/td_loss": sum(avg_loss) / max(len(avg_loss), 1)},
						                        step=(it + start_record_it))
				logger.info("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (epoch - episode_start, eps, episode_rewards))
				# Reset params that determine how foods are spawn
				env.set_objective(env.foods[rng_gen.integers(n_foods_spawn)].position)
				env.food_spawn_pos = None
				env.n_food_spawn = 0
				env.spawn_food(n_foods_spawn, env.max_food_level)
				obs, *_ = env.reset()
				done = True
				episode_rewards = 0
				episode_start = epoch


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
	
	# Train parameters
	parser.add_argument('--train-performance', dest='min_train_performance', type=float, default=MIN_TRAIN_PERFORMANCE, help='Minimum performance threshold to skip model train')
	parser.add_argument('--max-cycles', dest='max_cycles', type=int, required=True, help='Max number of training cycles.')
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
	parser.add_argument('--cycle-type', dest='cycle_type', type=str, required=False, default='log', choices=['linear', 'log'],
	                    help='Type of cycle eps update to use: linear, logarithmic (log)')
	parser.add_argument('--warmup-steps', dest='warmup', type=int, required=False, default=10000, help='Number of epochs to pass before training starts')
	parser.add_argument('--tensorboard-freq', dest='tensorboard_freq', type=int, required=False, default=1,
	                    help='Number of epochs between each log in tensorboard. Use only in combination with --tensorboard option')
	parser.add_argument('--restart', dest='restart_train', action='store_true',
	                    help='Flag that signals that train is suppose to restart from a previously saved point.')
	parser.add_argument('--restart-info', dest='restart_info', type=str, nargs='+', required=False, default=None,
	                    help='List with the info required to recover previously saved model and restart from same point: '
	                         '<model_dirname: str> <model_filename: str> <last_cycle: int> Use only in combination with --restart option')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag signalling debug mode for model training')
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--epoch-logging', dest='ep_log', action='store_true', help='')
	parser.add_argument('--train-tags', dest='tags', type=str, nargs='+', required=False, default=None,
	                    help='List of tags for grouping in weights and biases, empty by default signaling not to train under a specific set of tags')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='', help='Directory to store trained models, if left blank stored in default location')
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
	parser.add_argument('--player-level', dest='player_level', type=int, required=True, help='Level of the agents collecting food')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--n-food', dest='n_foods', type=int, required=True, help='Number of food items in the field')
	parser.add_argument('--food-level', dest='food_level', type=int, required=True, help='Level of the food items')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	parser.add_argument('--render', dest='use_render', action='store_true', help='Flag that signals the use of the field render while training')
	parser.add_argument('--n-foods-spawn', dest='n_foods_spawn', type=int, required=True, help='Number of foods to be spawned for training.')
	parser.add_argument('--no-force-coop', dest='no_force_coop', action='store_true', help='Flag denoting that the agents do not need to pick all items in full cooperation')
	
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
	use_tracker = args.use_tensorboard
	tracker_dir = args.tracker_dir
	
	# Train args
	train_thresh = args.min_train_performance
	n_iterations = args.n_iterations
	batch_size = args.batch_size
	train_freq = args.train_freq
	target_freq = args.target_freq
	online_lr = args.learn_rate
	target_lr = args.target_learn_rate
	initial_eps = args.initial_eps
	final_eps = args.final_eps
	eps_decay = args.eps_decay
	eps_type = args.eps_type
	warmup = args.warmup
	tracker_frq = args.tensorboard_freq
	restart_train = args.restart_train
	restart_info = args.restart_info
	debug = args.debug
	use_lower_model = args.use_lower_model
	use_higher_model = args.use_higher_model
	
	# LB-Foraging environment args
	player_level = args.player_level
	field_lengths = args.field_lengths
	n_foods = args.n_foods
	food_level = args.food_level
	max_steps = args.max_steps
	use_render = args.use_render
	n_foods_spawn = args.n_foods_spawn
	force_coop = not args.no_force_coop
	
	try:
		assert not (use_higher_model and use_lower_model)
	
	except AssertionError:
		print('Attempt at using curriculum learning using both model trained with one more and one less food item spawned')
		return
	
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
		logging.error('[ARGS ERROR] Field size must either be composed of only 1 or 2 arguments; %d were given. Exiting program' % field_dims)
		return
	
	now = datetime.now()
	home_dir = Path(__file__).parent.absolute().parent.absolute()
	log_dir = Path(args.logs_dir) if args.logs_dir != '' else home_dir / 'logs'
	data_dir = Path(args.data_dir) if args.data_dir != '' else home_dir / 'data'
	models_dir = Path(args.models_dir) if args.models_dir != '' else home_dir / 'models'
	if tracker_dir == '':
		tracker_dir = log_dir
	log_filename = (('train_lb_coop_single_v3_dqn_%dx%d-field_%d-agents_%d-foods_%d-food-level' % (field_size[0], field_size[1], n_agents,
	                                                                                               n_foods_spawn, food_level)) +
	                '_' + now.strftime("%Y%m%d-%H%M%S"))
	model_path = (models_dir / ('lb_coop%s_single%s_dqn' % ('_mixed' if not force_coop else '', '_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1])) /
	              ('%d-agents' % n_agents) / ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / now.strftime("%Y%m%d-%H%M%S"))
	
	with open(data_dir / 'performances' / 'lb_foraging' / ('train_performances%s_%sa%s_all_foods.yaml' % ('_vdn' if use_vdn else '', str(n_agents), '_mixed' if not force_coop else '')),
	          mode='r+', encoding='utf-8') as train_file:
		train_performances = yaml.safe_load(train_file)
		field_idx = str(field_size[0]) + 'x' + str(field_size[1])
		food_idx = str(n_foods_spawn) + '-food'
		train_acc = train_performances[field_idx][food_idx]
	
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.safe_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = [tuple(x) for x in config_params['food_locs'][dict_idx]]
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
	
	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)
	
	logging.basicConfig(filename=(log_dir / (log_filename + '_log.txt')), filemode='w', format='%(name)s %(asctime)s %(levelname)s:\t%(message)s',
	                    level=logging.INFO)
	logger = logging.getLogger('INFO')
	err_logger = logging.getLogger('ERROR')
	handler = logging.StreamHandler(sys.stderr)
	handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
	err_logger.addHandler(handler)
	Path.mkdir(model_path, parents=True, exist_ok=True)
	
	logger.info('##############################')
	logger.info('Starting LB Foraging DQN Train')
	logger.info('##############################')
	gc.enable()
	
	####################
	## Training Model ##
	####################
	if train_acc < train_thresh:
		try:
			wandb_run = wandb.init(
					project='lb-foraging-optimal', entity='miguel-faria',
					config={
							"field": "%dx%d" % (field_size[0], field_size[1]),
							"agents": n_agents,
							"foods": n_foods,
							"online_learing_rate": online_lr,
							"target_learning_rate": target_lr,
							"discount": gamma,
							"eps_decay": eps_type,
							"eps_rate": eps_decay,
							"dqn_architecture": architecture,
							"iterations": n_iterations,
							"cycles": 1,
							"buffer_size": buffer_size,
							"buffer_add": "smart" if args.buffer_smart_add else "plain",
							"buffer_add_method": args.buffer_method if args.buffer_smart_add else "fifo",
							"batch_size": batch_size,
							"curriculum_learning": 'no' if not (use_higher_model or use_lower_model) else ('lower_model' if use_lower_model else 'higher_model'),
							"full_cooperation": force_coop
					},
					dir=tracker_dir,
					name=('%s%ssingle_v3-l%dx%d-%df-' % ('mixed-' if not force_coop else '', 'vdn-' if use_vdn else 'independent-', field_size[0], field_size[1], n_foods_spawn) +
					      now.strftime("%Y%m%d-%H%M%S")),
					sync_tensorboard=True)
			logger.info('Starting training model location agnostic')
			logger.info('Environment setup')
			env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, force_coop, food_level, RNG_SEED, food_locs,
									 use_encoding=True, agent_center=True, grid_observation=use_cnn)
			env.seed(RNG_SEED)
			logger.info('Setup multi-agent DQN')
			agent_action_space = env.action_space[0]
			if isinstance(env.observation_space, MultiBinary):
				obs_space = MultiBinary([*env.observation_space.shape[1:]])
			else:
				obs_space = env.observation_space[0]
			if use_vdn:
				action_space = MultiDiscrete([agent_action_space.n] * env.n_players)
				agent_madqn = SingleModelMADQN(n_agents, agent_action_space.n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, action_space, env.observation_space, use_gpu,
											   dueling_dqn, use_ddqn, use_vdn, use_cnn, False, cnn_properties=cnn_properties,
											   buffer_data=(args.buffer_smart_add, args.buffer_method))
			else:
				agent_madqn = SingleModelMADQN(n_agents, agent_action_space.n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, agent_action_space, obs_space, use_gpu,
											   dueling_dqn, use_ddqn, use_vdn, use_cnn, False, cnn_properties=cnn_properties,
											   buffer_data=(args.buffer_smart_add, args.buffer_method))
			if restart_train:
				logger.info('Load trained model')
				agent_madqn.load_model(restart_info[1], model_path.parent.absolute() / restart_info[0], logger,
									   env.observation_space[0].shape if not use_cnn else (1, *env.observation_space[0].shape))
			else:
				logger.info('Starting train')
			
			if use_lower_model and n_foods_spawn > 1:
				prev_model_path = model_path.parent.parent.absolute() / ('%d-foods_%d-food-level' % (max(n_foods_spawn - 1, 1), food_level)) / 'best'
				if (prev_model_path / 'all_foods_single_model.model').exists():
					logger.info('Using model trained with %d foods spawned as a baseline' % (max(n_foods_spawn - 1, 1)))
					curriculum_model_path = str(prev_model_path / 'all_foods_single_model.model')
				else:
					logger.info('Model with one less food item not found, training from scratch')
					curriculum_model_path = ''
			elif use_higher_model and n_foods_spawn < n_foods:
				next_model_path = model_path.parent.parent.absolute() / ('%d-foods_%d-food-level' % (min(n_foods_spawn + 1, n_foods), food_level)) / 'best'
				if (next_model_path / 'all_foods_single_model.model').exists():
					logger.info('Using model trained with %d foods spawned as a baseline' % (min(n_foods_spawn + 1, n_foods)))
					curriculum_model_path = str(next_model_path / 'all_foods_single_model.model')
				else:
					logger.info('Model with one more food item not found, training from scratch')
					curriculum_model_path = ''
			else:
				logger.info('Training model from scratch')
				curriculum_model_path = ''
			
			logger.info('Starting train')
			cnn_shape = (0, ) if not agent_madqn.agent_dqn.cnn_layer else (*obs_space.shape[1:], obs_space.shape[0])
			tracker_panel = 'l%dx%d-%df' % (field_size[0], field_size[1], n_foods_spawn)
			greedy_actions = False
			train_lb_model(env, agent_madqn, n_iterations, max_steps * n_iterations, n_foods_spawn, batch_size, online_lr, target_lr, initial_eps, final_eps, eps_type,
						   RNG_SEED, logger, cnn_shape, eps_decay, warmup, train_freq, target_freq, tracker_frq, use_render, greedy_actions, args.ep_log,
						   curriculum_model_path, use_tracker, wandb_run, tracker_panel, debug)
			
			env.close()
			logger.info('Saving final model')
			agent_madqn.save_model('all_foods', model_path, logger)
				
			####################
			## Testing Model ##
			####################
			logger.info('Testing trained model')
			env = FoodCOOPLBForaging(n_agents, player_level, field_size, n_foods, sight, max_steps, True, food_level, TEST_RNG_SEED, food_locs,
									 use_encoding=True, agent_center=True, grid_observation=use_cnn)
			if isinstance(env.observation_space, MultiBinary):
				obs_space = MultiBinary([*env.observation_space.shape[1:]])
			else:
				obs_space = env.observation_space[0]
			cnn_shape = (0, ) if not agent_madqn.agent_dqn.cnn_layer else (*obs_space.shape[1:], obs_space.shape[0])
			tests_passed = 0
			env.seed(TEST_RNG_SEED)
			np.random.seed(TEST_RNG_SEED)
			rng_gen = np.random.default_rng(TEST_RNG_SEED)
			avg_nr_epochs = []
			for i in range(N_TESTS):
				env.spawn_players([player_level] * n_agents)
				env.spawn_food(n_foods_spawn, food_level)
				logger.info('Test number %d' % (i + 1))
				logger.info('Agents: ' + ', '.join(['%s @ (%d, %d) with level %d' % (player.player_id, *player.position, player.level) for player in env.players]))
				logger.info('Number of food spawn:\t%d' % n_foods_spawn)
				logger.info('Food items: ' + ', '.join(['(%d, %d) with level: %d' % (*food.position, food.level) for food in env.foods]))
				logger.info('Food objective: (%d, %d)' % env.obj_food)
				obs, *_ = env.reset()
				epoch = 0
				agent_reward = [0] * n_agents
				game_over = False
				finished = False
				timeout = False
				while not game_over:
					
					actions = []
					for a_idx in range(agent_madqn.num_agents):
						dqn = agent_madqn.agent_dqn
						if use_cnn:
							cnn_obs = obs[a_idx].reshape((1, *cnn_shape))
							q_values = dqn.q_network.apply(dqn.online_state.params, cnn_obs)[0]
						else:
							q_values = dqn.q_network.apply(dqn.online_state.params, obs[a_idx])
						pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
						pol = pol / pol.sum()
						action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						action = jax.device_get(action)
						actions += [action]
					actions = np.array(actions)
					next_obs, rewards, finished, timeout, infos = env.step(actions)
					logger.info(env.get_env_log() + 'Actions: ' + str([Action(act).name for act in actions]))
					agent_reward = [agent_reward[idx] + rewards[idx] for idx in range(n_agents)]
					obs = next_obs
					
					if finished or timeout:
						game_over = True
						loc = env.foods[rng_gen.integers(n_foods_spawn)].position
						env.set_objective(loc)
						env.food_spawn_pos = None
						env.n_food_spawn = 0
					
					else:
						epoch += 1
					
					sys.stdout.flush()
				
				if finished:
					tests_passed += 1
					avg_nr_epochs += [epoch]
					logger.info('Test %d finished in success' % (i + 1))
					logger.info('Number of epochs: %d' % epoch)
					logger.info('Accumulated reward:\n\t' + '\n\t'.join(['- agent %d: %.2f' % (idx + 1, agent_reward[idx]) for idx in range(n_agents)]))
					logger.info('Average reward:\n\t' + '\n\t'.join(['- agent %d: %.2f' % (idx + 1, agent_reward[idx] / epoch) for idx in range(n_agents)]))
				if timeout:
					logger.info('Test %d timed out' % (i + 1))
					avg_nr_epochs += [epoch]
			
			env.close()
			logger.info('Passed %d tests out of %d' % (tests_passed, N_TESTS))
			logger.info('Average number of steps per test: %d' % np.mean(avg_nr_epochs))
			
			if (tests_passed / N_TESTS) > train_acc:
				logger.info('Updating best model for current loc')
				Path.mkdir(model_path.parent.absolute() / 'best', parents=True, exist_ok=True)
				agent_madqn.save_model('all_foods', model_path.parent.absolute() / 'best', logger)
				train_acc = tests_passed / N_TESTS
			
			agent_madqn = None
			del env
			del agent_madqn
			gc.collect()
		
			logger.info('Updating best training performances record with %dx%d results' % (loc[0], loc[1]))
			with open(data_dir / 'performances' / 'lb_foraging' / ('train_performances%s_%sa%s_all_foods.yaml' % ('_vdn' if use_vdn else '', str(n_agents),
																												  '_mixed' if not force_coop else '')),
					  mode='r+', encoding='utf-8') as train_file:
				performance_data = yaml.safe_load(train_file)
				field_idx = str(field_size[0]) + 'x' + str(field_size[1])
				food_idx = str(n_foods_spawn) + '-food'
				performance_data[field_idx][food_idx] = train_acc
				train_file.seek(0)
				sorted_data = dict(
						[[sorted_key, performance_data[sorted_key]] for sorted_key in
						 [str(t[0]) + 'x' + str(t[1]) for t in sorted([tuple([int(x) for x in key.split('x')]) for key in performance_data.keys()])]])
				yaml.safe_dump(sorted_data, train_file)
		
			wandb_run.finish()
		
		except KeyboardInterrupt as ks:
			logger.info('Caught keyboard interrupt, cleaning up and closing.')
			with open(data_dir / 'performances' / 'lb_foraging' / ('train_performances%s_%sa%s_all_foods.yaml' % ('_vdn' if use_vdn else '', str(n_agents),
																												  '_mixed' if not force_coop else '')),
			          mode='r+', encoding='utf-8') as train_file:
				performance_data = yaml.safe_load(train_file)
				field_idx = str(field_size[0]) + 'x' + str(field_size[1])
				food_idx = str(n_foods_spawn) + '-food'
				performance_data[field_idx][food_idx] = train_acc
				train_file.seek(0)
				sorted_data = dict(
						[[sorted_key, performance_data[sorted_key]] for sorted_key in
						 [str(t[0]) + 'x' + str(t[1]) for t in sorted([tuple([int(x) for x in key.split('x')]) for key in performance_data.keys()])]])
				yaml.safe_dump(sorted_data, train_file)
				
		finally:
			wandb.finish()


if __name__ == '__main__':
	main()
