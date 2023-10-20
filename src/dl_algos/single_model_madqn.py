#! /usr/bin/env python
import sys
import time
import flax
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
import logging

from flax.training.train_state import TrainState
from gymnasium.spaces import Space
from pathlib import Path
from dl_algos.dqn import DQNetwork, EPS_TYPE
from typing import List, Dict, Callable, Tuple
from datetime import datetime


# noinspection DuplicatedCode,PyTypeChecker
class SingleModelMADQN(object):
	
	_num_agents: int
	_agent_dqn: DQNetwork
	_write_tensorboard: bool
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_cnn: bool = False, handle_timeout: bool = False,
				 use_tensorboard: bool = False, tensorboard_data: List = None, cnn_properties: List[int] = None):
		
		"""
		Initialize a multi-agent scenario DQN with a single DQN model

		:param num_agents: number of agents in the environment
		:param action_dim: number of actions of the agent, the DQN is agnostic to the semantic of each action
        :param num_layers: number of layers for the q_network
        :param act_function: activation function for the q_network
        :param layer_sizes: number of neurons in each layer (list must have equal number of entries as the number of layers)
        :param buffer_size: buffer size for the replay buffer
        :param gamma: reward discount factor
        :param observation_space: gym space for the agent observations
        :param use_gpu: flag that controls use of cpu or gpu
        :param handle_timeout: flag that controls handle timeout termination (due to timelimit) separately and treat the task as infinite horizon task.
        :param use_tensorboard: flag that notes usage of a tensorboard summary writer (default: False)
        :param tensorboard_data: list of the form [log_dir: str, queue_size: int, flush_interval: int, filename_suffix: str] with summary data for
        the summary writer (default is None)

        :type num_agents: int
        :type action_dim: int
        :type num_layers: int
        :type buffer_size: int
        :type layer_sizes: list[int]
        :type use_gpu: bool
        :type handle_timeout: bool
        :type use_tensorboard: bool
        :type gamma: float
        :type act_function: callable
        :type observation_space: gym.Space
        :type tensorboard_data: list
		"""
		
		self._num_agents = num_agents
		self._write_tensorboard = use_tensorboard
		now = datetime.now()
		if tensorboard_data is not None:
			if len(tensorboard_data) == 4:
				board_data = [tensorboard_data[0] + '/single_model_' + now.strftime("%Y%m%d-%H%M%S"), tensorboard_data[1], tensorboard_data[2],
							  tensorboard_data[3], 'central_train']
			else:
				board_data = [tensorboard_data[0] + '/single_model_' + now.strftime("%Y%m%d-%H%M%S") + '_' + tensorboard_data[4], tensorboard_data[1],
							  tensorboard_data[2], tensorboard_data[3], 'central_train']
		else:
			board_data = tensorboard_data
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, num_agents * buffer_size, gamma, observation_space, use_gpu, dueling_dqn,
									use_ddqn, use_cnn, handle_timeout, use_tensorboard, board_data, cnn_properties)
	
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._num_agents
	
	@property
	def agent_dqn(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@num_agents.setter
	def num_agents(self, num_agents: int):
		self._num_agents = num_agents
	
	@agent_dqn.setter
	def agent_dqn(self, agent_dqn: DQNetwork):
		self._agent_dqn = agent_dqn
	
	@write_tensorboard.setter
	def write_tensorboard(self, write_tensorboard: bool):
		self._write_tensorboard = write_tensorboard
	
	#####################
	### Class Methods ###
	#####################
	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1,
				  target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0, greedy_action: bool = True):
		
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		if not self._agent_dqn.dqn_initialized:
			if self._agent_dqn.cnn_layer:
				self._agent_dqn.init_network_states(rng_seed, obs[0].reshape((1, *obs[0].shape)), optim_learn_rate)
			else:
				self._agent_dqn.init_network_states(rng_seed, obs[0], optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_start = epoch
			episode_history = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				if rng_gen.random() < eps: # Exploration
					actions = np.array(env.action_space.sample())
				else: # Exploitation
					actions = []
					for a_idx in range(self._num_agents):
						# Compute q_values
						if self._agent_dqn.cnn_layer:
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
						else:
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx])
						# Get action
						if greedy_action:
							action = q_values.argmax(axis=-1)
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						action = jax.device_get(action)
						if self._agent_dqn.use_summary and epoch % tensorboard_frequency == 0:
							self._agent_dqn.summary_writer.add_scalar("charts/episodic_q_vals", float(q_values[int(action)]), epoch + start_record_epoch)
						actions += [action]
					actions = np.array(actions)
				episode_history += [self.get_history_entry(obs, actions)]
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				if use_render:
					env.render()
				
				if len(rewards) == 1:
					rewards = np.array([rewards] * self._num_agents)
				
				if terminated:
					finished = np.ones(self._num_agents)
				else:
					finished = np.zeros(self._num_agents)
				
				# store new samples
				real_next_obs = list(next_obs).copy()
				for a_idx in range(self._num_agents):
					self._agent_dqn.replay_buffer.add(obs[a_idx], real_next_obs[a_idx], actions[a_idx], rewards[a_idx], finished[a_idx], infos)
					episode_rewards += (rewards[a_idx] / self._num_agents)
				if self._agent_dqn.use_summary:
					self._agent_dqn.summary_writer.add_scalar("charts/reward", sum(rewards) / self._num_agents, epoch + start_record_epoch)
				obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						self.update_model(batch_size, epoch, start_time, tensorboard_frequency, logger)
					
					if epoch % target_freq == 0:
						self._agent_dqn.update_target_model(tau)
				
				epoch += 1
				sys.stdout.flush()
				
				# Check if iteration is over
				if terminated or timeout:
					if self._write_tensorboard:
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					logger.debug("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (epoch - episode_start, eps, episode_rewards))
					obs, *_ = env.reset()
					done = True
					history += [episode_history]
					episode_rewards = 0
					episode_start = epoch
		
		return history
	
	def update_model(self, batch_size, epoch, start_time, tensorboard_frequency, logger: logging.Logger):
		train_info = ('epoch: %d \t' % epoch)
		losses = []
		for a_idx in range(self._num_agents):
			loss = self._agent_dqn.update_online_model(batch_size, epoch, start_time, tensorboard_frequency)
			losses += [loss]
			train_info += ('agent %d: loss: %.7f\t' % (a_idx, loss))
		logger.debug('Train Info: ' + train_info)
		return sum(losses) / self._num_agents
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		model_path = model_dir / (filename + '_single_model.model')
		if logger.level == logging.DEBUG:
			params_shapes = ''
			for key in self._agent_dqn.online_state.params.keys():
				if isinstance(self._agent_dqn.online_state.params[key], flax.core.FrozenDict):
					for key2 in self._agent_dqn.online_state.params[key].keys():
						if isinstance(self._agent_dqn.online_state.params[key][key2], flax.core.FrozenDict):
							for key3 in self._agent_dqn.online_state.params[key][key2].keys():
								params_shapes += ('%s: %s ' % (key3, ', '.join([str(x) for x in self._agent_dqn.online_state.params[key][key2][key3].shape])))
						else:
							params_shapes += ('%s: %s ' % (key2, ', '.join([str(x) for x in self._agent_dqn.online_state.params[key][key2].shape])))
				else:
					params_shapes += ('%s: %s ' % (key, ', '.join([str(x) for x in self._agent_dqn.online_state.params[key].shape])))
			logger.debug('Model params: %s' % params_shapes)
		with open(model_path, "wb") as mf:
			mf.write(flax.serialization.to_bytes(self._agent_dqn.online_state))
		logger.info("Model state saved to file: " + str(model_path))
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: Tuple) -> None:
		file_path = model_dir / (filename + '_single_model.model')
		template = TrainState.create(apply_fn=self._agent_dqn.q_network.apply,
									 params=self._agent_dqn.q_network.init(jax.random.PRNGKey(201), jnp.empty(obs_shape)),
									 tx=optax.adam(learning_rate=0.0001))
		with open(file_path, "rb") as f:
			self._agent_dqn.online_state = flax.serialization.from_bytes(template, f.read())
		logger.info("Loaded model state from file: " + str(file_path))
	
	def get_history_entry(self, obs: np.ndarray, actions: List):
		
		entry = []
		for idx in range(self._num_agents):
			entry += [' '.join([str(x) for x in obs[idx]]), str(actions[idx])]
		
		return entry


# noinspection PyTypeChecker,DuplicatedCode
class LegibleCentralMADQN(SingleModelMADQN):
	_optimal_models: Dict[str, TrainState]
	_goal_ids: List[str]
	_goal: str
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 observation_space: Space, use_gpu: bool, handle_timeout: bool, models_dir: Path, optimal_filenames: List[str], goal_ids: List[str], goal: str,
				 dueling_dqn: bool = False, use_ddqn: bool = False, use_cnn: bool = False, use_tensorboard: bool = False, tensorboard_data: List = None):
		
		super().__init__(num_agents, action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, observation_space, use_gpu,dueling_dqn, use_ddqn,
						 use_cnn, handle_timeout, use_tensorboard, tensorboard_data)
		
		self._goal_ids = goal_ids.copy()
		self._goal = goal
		for goal_id in goal_ids:
			idx = goal_ids.index(goal_id)
			file_path = models_dir / optimal_filenames[idx]
			obs_shape = (1, *observation_space.shape) if use_cnn else observation_space.shape
			template = TrainState.create(apply_fn=self._agent_dqn.q_network.apply,
										 params=self._agent_dqn.q_network.init(jax.random.PRNGKey(201), jnp.empty(obs_shape), train=False),
										 tx=optax.adam(learning_rate=0.0001))
			with open(file_path, "rb") as f:
				self._optimal_models[goal_id] = flax.serialization.from_bytes(template, f.read())
	
	########################
	### Class Properties ###
	########################
	@property
	def optimal_models(self):
		return self._optimal_models
	
	def optimal_model(self, goal: str):
		return self._optimal_models[goal]
	
	@property
	def goal_ids(self):
		return self._goal_ids
	
	#####################
	### Class Methods ###
	#####################
	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1,
				  target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0, greedy_action: bool = True):
		
		rng_gen = np.random.default_rng(rng_seed)
		
		# Setup DQNs for training
		obs, _ = env.reset()
		self._agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		history = []
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
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
					actions = np.array(env.action_space.sample())
				else:
					actions = []
					for a_idx in range(self._num_agents):
						if self._agent_dqn.cnn_layer:
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
						else:
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx])
						if greedy_action:
							action = q_values.argmax(axis=-1)
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						action = jax.device_get(action)
						actions += [action]
					actions = np.array(actions)
				next_obs, _, terminated, timeout, infos = env.step(actions)
				episode_history += [self.get_history_entry(obs, actions)]
				if use_render:
					env.render()
				
				# Obtain the legible rewards
				legible_rewards = []
				for a_idx in range(self._num_agents):
					act_q_vals = []
					goal_q = 0.0
					action = actions[a_idx]
					for goal in self._goal_ids:
						act_q_vals += [self._agent_dqn.q_network.apply(self._optimal_models[goal].params, obs[a_idx])[action]]
						if goal == self._goal:
							goal_q = act_q_vals[-1]
					legible_rewards += [goal_q / sum(act_q_vals)]
				legible_rewards = np.array(legible_rewards)
				
				if terminated:
					finished = np.ones(self._num_agents)
				else:
					finished = np.zeros(self._num_agents)
				
				# store new samples
				real_next_obs = list(next_obs).copy()
				for a_idx in range(self._num_agents):
					self._agent_dqn.replay_buffer.add(obs[a_idx], real_next_obs[a_idx], actions[a_idx], legible_rewards[a_idx], finished[a_idx], infos)
				obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						self.update_model(batch_size, epoch, start_time, tensorboard_frequency, logger)
					
					if epoch % target_freq == 0:
						self._agent_dqn.update_target_model(tau)
				
				epoch += 1
				sys.stdout.flush()
				
				# Check if iteration is over
				if terminated or timeout:
					if self._write_tensorboard:
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					logger.debug("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (epoch - episode_start, eps, episode_rewards))
					obs, *_ = env.reset()
					done = True
					history += [episode_history]
					episode_rewards = 0
					episode_start = epoch
		
		return history


# noinspection PyTypeChecker,DuplicatedCode,PyUnresolvedReferences
class CentralizedMADQN(object):
	
	_num_agents: int
	_madqn: DQNetwork
	_write_tensorboard: bool
	_joint_action_converter: Callable
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_converter: Callable, act_function: Callable, layer_sizes: List[int],
				 buffer_size: int, gamma: float, observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False,
				 use_cnn: bool = False, handle_timeout: bool = False, use_tensorboard: bool = False, tensorboard_data: List = None):
		
		self._num_agents = num_agents
		self._write_tensorboard = use_tensorboard
		self._joint_action_converter = act_converter
		now = datetime.now()
		if tensorboard_data is not None:
			board_data = [tensorboard_data[0] + '/centralized_madqn_' + now.strftime("%Y%m%d-%H%M%S"), tensorboard_data[1], tensorboard_data[2],
						  tensorboard_data[3], 'centralized_madqn']
		else:
			board_data = tensorboard_data
		dqn_action_dim = action_dim ** num_agents
		self._madqn = DQNetwork(dqn_action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, observation_space, use_gpu,
								dueling_dqn, use_ddqn, use_cnn, handle_timeout, use_tensorboard, board_data)
		
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._num_agents
	
	@property
	def madqn(self) -> DQNetwork:
		return self._madqn
	
	@property
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@property
	def get_joint_action(self) -> Callable:
		return self._joint_action_converter
	
	@num_agents.setter
	def num_agents(self, num_agents: int):
		self._num_agents = num_agents
	
	@madqn.setter
	def madqn(self, agent_dqn: DQNetwork):
		self._madqn = agent_dqn
	
	@write_tensorboard.setter
	def write_tensorboard(self, write_tensorboard: bool):
		self._write_tensorboard = write_tensorboard
		
	#####################
	### Class Methods ###
	#####################
	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float,
				  initial_eps: float, final_eps: float, eps_type: str, rng_seed: int, log_filename: str, exploration_decay: float = 0.99, warmup: int = 0,
				  train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0):
		
		rng_gen = np.random.default_rng(rng_seed)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		joint_obs = np.array(obs).ravel()
		self._madqn.init_network_states(rng_seed, joint_obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_start = epoch
			episode_history = []
			print("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				
				# interact with environment
				joint_obs = np.array(obs).ravel()
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				if rng_gen.random() < eps:
					joint_action = np.array(rng_gen.choice(range(self._madqn.q_network.action_dim)))
				else:
					q_values = self._madqn.q_network.apply(self._madqn.online_state.params, joint_obs)
					joint_action = q_values.argmax(axis=-1)
					joint_action = jax.device_get(joint_action)
					if self._write_tensorboard:
						self._madqn.summary_writer.add_scalar("charts/episodic_q_vals", float(q_values[int(joint_action)]), epoch + start_record_epoch)
				actions = self._joint_action_converter(joint_action, self._num_agents)
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				episode_history += [self.get_history_entry(obs, actions)]
				if use_render:
					env.render()
				
				if len(rewards) == 1:
					rewards = np.array([rewards] * self._num_agents)
				
				if terminated:
					finished = np.ones(self._num_agents)
				else:
					finished = np.zeros(self._num_agents)
				
				# store new samples
				joint_next_obs = np.array(next_obs).ravel()
				self._madqn.replay_buffer.add(joint_obs, joint_next_obs, joint_action, sum(rewards) / self._num_agents, finished[0], infos)
				for a_idx in range(self._num_agents):
					episode_rewards += (rewards[a_idx] / self._num_agents)
					if self._write_tensorboard:
						self._madqn.summary_writer.add_scalar("charts/reward", rewards[a_idx], epoch + start_record_epoch)
				obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						loss = self.update_model(batch_size, epoch, start_time, tensorboard_frequency)
					
					if epoch % target_freq == 0:
						self._madqn.update_target_model(tau)
				
				epoch += 1
				sys.stdout.flush()
				
				# Check if iteration is over
				if terminated or timeout:
					if self._write_tensorboard:
						self._madqn.summary_writer.add_scalar("charts/episodic_return", episode_rewards, it + start_record_it)
						self._madqn.summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it + start_record_it)
						self._madqn.summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
					obs, *_ = env.reset()
					done = True
					history += [episode_history]
					episode_rewards = 0
					episode_start = epoch
		
		return history
	
	def update_model(self, batch_size, epoch, start_time, tensorboard_frequency):
		train_info = ('epoch: %d \t' % epoch)
		losses = []
		for a_idx in range(self._num_agents):
			loss = self._madqn.update_online_model(batch_size, epoch, start_time, tensorboard_frequency)
			losses += [loss]
			train_info += ('agent %d: loss: %.7f\t' % (a_idx, loss))
		# print('Train Info: ' + train_info)
		return sum(losses) / self._num_agents
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		model_path = model_dir / (filename + '_ctce.model')
		with open(model_path, "wb") as mf:
			mf.write(flax.serialization.to_bytes(self._madqn.online_state))
		logger.info("Model state saved to file: " + str(model_path))
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: Tuple) -> None:
		file_path = model_dir / (filename + '_ctce.model')
		template = TrainState.create(apply_fn=self._madqn.q_network.apply,
									 params=self._madqn.q_network.init(jax.random.PRNGKey(201), jnp.empty(obs_shape)),
									 tx=optax.adam(learning_rate=0.0001))
		with open(file_path, "rb") as f:
			self._madqn.online_state = flax.serialization.from_bytes(template, f.read())
		logger.info("Loaded model state from file: " + str(file_path))
	
	def get_history_entry(self, obs: np.ndarray, actions: List):
		
		entry = []
		for idx in range(self._num_agents):
			entry += [' '.join([str(x) for x in obs[idx]]), str(actions[idx])]
		
		return entry