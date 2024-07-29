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
from dl_utilities.buffers import ReplayBuffer
from typing import List, Dict, Callable, Tuple, Optional
from datetime import datetime
from functools import partial
from jax import jit
from wandb.wandb_run import Run


# noinspection DuplicatedCode,PyTypeChecker
class SingleModelMADQN(object):
	
	_n_agents: int
	_agent_dqn: DQNetwork
	_replay_buffer: ReplayBuffer
	_use_vdn: bool
	_use_ddqn: bool
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float, action_space: Space,
				 observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_vdn: bool = False, use_cnn: bool = False,
				 handle_timeout: bool = False, cnn_properties: List = None, n_envs: int = 1, buffer_data: tuple = (False, '')):
		
		"""
		Initialize a multi-agent scenario DQN with a single DQN model

		:param num_agents: 			number of agents in the environment
		:param action_dim: 			number of actions of the agent, the DQN is agnostic to the semantic of each action
        :param num_layers: 			number of layers for the q_network
        :param act_function: 		activation function for the q_network
        :param layer_sizes: 		number of neurons in each layer (list must have equal number of entries as the number of layers)
        :param buffer_size: 		buffer size for the replay buffer
        :param gamma: 				reward discount factor
        :param observation_space: 	gym space for the agent observations
        :param use_gpu: 			flag that controls use of cpu or gpu
        :param handle_timeout: 		flag that controls handle timeout termination (due to timelimit) separately and treat the task as infinite horizon task.

        :type num_agents: int
        :type action_dim: int
        :type num_layers: int
        :type buffer_size: int
        :type layer_sizes: list[int]
        :type use_gpu: bool
        :type handle_timeout: bool
        :type gamma: float
        :type act_function: callable
        :type observation_space: gym.Space
		"""
		
		self._n_agents = num_agents
		self._use_vdn = use_vdn
		self._use_ddqn = use_ddqn
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties)
		if use_vdn:
			self._replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu", handle_timeout_termination=handle_timeout,
											   n_agents=num_agents, n_envs=n_envs, smart_add=buffer_data[0], add_method=buffer_data[1])
		else:
			self._replay_buffer = ReplayBuffer(buffer_size * num_agents, observation_space, action_space, "cuda" if use_gpu else "cpu", handle_timeout_termination=handle_timeout,
											   n_envs=n_envs, smart_add=buffer_data[0], add_method=buffer_data[1])
		
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._n_agents
	
	@property
	def agent_dqn(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
	
	@property
	def use_vdn(self) -> bool:
		return self._use_vdn
	
	#####################
	### Class Methods ###
	#####################
	def mse_loss(self, params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray, next_q_value: jnp.ndarray):
		q = jnp.zeros((next_q_value.shape[0]))
		for idx in range(self._n_agents):
			qa = self._agent_dqn.q_network.apply(params, observations[:, idx])
			q += qa[np.arange(qa.shape[0]), actions[:, idx].squeeze()]
		q = q.reshape(-1, 1)
		return ((q - next_q_value) ** 2).mean(), q
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_dqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							  next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros(n_obs)
		for idx in range(self._n_agents):
			next_q_value += self._agent_dqn.compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx], target_state_params)
		# next_q_value = next_q_value / n_agents
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_pred, q_state
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_ddqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							  next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros((n_obs, 1))
		for idx in range(self._n_agents):
			next_q_value += self._agent_dqn.compute_ddqn_targets(dones, next_observations[:, idx], rewards[:, idx].reshape(-1, 1),
																 target_state_params, q_state.params)
		# next_q_value = next_q_value / n_agents
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_pred, q_state
	
	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, cnn_shape: Tuple[int], exploration_decay: float = 0.99, warmup: int = 0,
				  train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0, greedy_action: bool = True,
				  epoch_logging: bool = False, initial_model_path: str = '', use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = ''):
		
		rng_gen = np.random.default_rng(rng_seed)
		# self._replay_buffer.reseed(rng_seed)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		self.initialize_network(cnn_shape, logger, obs, optim_learn_rate, rng_seed, initial_model_path)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		avg_episode_len = []
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			episode_start = epoch
			avg_loss = []
			episode_history = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				explore = rng_gen.random() < eps
				if explore: # Exploration
					actions = np.array(env.action_space.sample())
				else: # Exploitation
					actions = []
					for a_idx in range(self._n_agents):
						# Compute q_values
						if self._agent_dqn.cnn_layer:
							cnn_obs = obs[a_idx].reshape((1, *cnn_shape))
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, cnn_obs)[0]
						else:
							q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx])
						# Get action
						if greedy_action:
							action = q_values.argmax()
						else:
							pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
							pol = pol / pol.sum()
							action = rng_gen.choice(range(env.action_space[0].n), p=pol)
						# action = jax.device_get(action)
						episode_q_vals += (float(q_values[int(action)]) / self._n_agents)
						actions += [action]
					actions = np.array(actions)
				if not self._agent_dqn.cnn_layer:
					episode_history += [self.get_history_entry(obs, actions)]
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				if use_render:
					env.render()
				
				if len(rewards) == 1:
					rewards = np.array([rewards] * self._n_agents)
				
				if terminated:
					finished = np.ones(self._n_agents)
				else:
					finished = np.zeros(self._n_agents)
				
				# store new samples
				if self.use_vdn:
					self.replay_buffer.add(obs, next_obs, actions, rewards, finished[0], infos)
					episode_rewards += sum(rewards) / self._n_agents
				else:
					for a_idx in range(self._n_agents):
						self.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], rewards[a_idx], finished[a_idx], infos)
						episode_rewards += (rewards[a_idx] / self._n_agents)
				if use_tracker and epoch_logging:
					performance_tracker.log({tracker_panel + "-charts/performance/reward": sum(rewards)}, step=(epoch + start_record_epoch))
				obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						loss = jax.device_get(self.update_model(batch_size, epoch, start_time, tensorboard_frequency, logger, cnn_shape))
						if use_tracker and epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss": loss}, step=epoch)
						else:
							avg_loss += [loss]
					
					if epoch % target_freq == 0:
						self._agent_dqn.update_target_model(tau)
				
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
								tracker_panel + "-charts/performance/episodic_length": episode_len,
								tracker_panel + "-charts/performance/avg_episode_length": np.mean(avg_episode_len),
								tracker_panel + "-charts/control/iteration": it,
								tracker_panel + "-charts/control/cycle": cycle,
								tracker_panel + "-charts/control/exploration": eps,
						},
								step=(it + start_record_it))
						if not epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss" : sum(avg_loss) / max(len(avg_loss), 1)},
																	step=(it + start_record_it))
					logger.info("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (epoch - episode_start, eps, episode_rewards))
					obs, *_ = env.reset()
					done = True
					history += [episode_history]
					episode_rewards = 0
					episode_start = epoch
		
		avg_loss = None
		return history
	
	def initialize_network(self, cnn_shape: Tuple, logger: logging.Logger, obs: np.ndarray, optim_learn_rate: float, rng_seed: int, previous_model_path: str = ''):
		if not self._agent_dqn.dqn_initialized:
			logger.info('Initializing network')
			if self._agent_dqn.cnn_layer:
				cnn_obs = obs[0].reshape((1, *cnn_shape))
				self._agent_dqn.init_network_states(rng_seed, cnn_obs, optim_learn_rate, previous_model_path)
			else:
				self._agent_dqn.init_network_states(rng_seed, obs[0], optim_learn_rate, previous_model_path)
	
	def update_model(self, batch_size, epoch, start_time, tensorboard_frequency, logger: logging.Logger, cnn_shape: Tuple[int] = None):
		data = self._replay_buffer.sample(batch_size)
		observations = data.observations
		next_observations = data.next_observations
		actions = data.actions
		rewards = data.rewards
		dones = data.dones
		
		if self._use_vdn:
			if self._agent_dqn.cnn_layer and cnn_shape is not None:
				observations = observations.reshape((*observations.shape[:2], *cnn_shape))
				next_observations = next_observations.reshape((*next_observations.shape[:2], *cnn_shape))
				
			if self._use_ddqn:
				loss, q_pred, self._agent_dqn.online_state = self.compute_vdn_ddqn_loss(self._agent_dqn.online_state, self._agent_dqn.target_params,
																						observations, actions, next_observations, rewards, dones)
			else:
				loss, q_pred, self._agent_dqn.online_state = self.compute_vdn_dqn_loss(self._agent_dqn.online_state, self._agent_dqn.target_params,
																					   observations, actions, next_observations, rewards, dones)
			loss = float(loss)
		else:
			if self._agent_dqn.cnn_layer and cnn_shape is not None:
				observations = observations.reshape((len(observations), *cnn_shape))
				next_observations = next_observations.reshape((len(next_observations), *cnn_shape))
			loss = self._agent_dqn.update_online_model(observations, actions, next_observations, rewards, dones)
		
		return loss
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqn.save_model(filename + '_single_model', model_dir, logger)
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: Tuple) -> None:
		self._agent_dqn.load_model(filename, model_dir, logger, obs_shape)
	
	def get_history_entry(self, obs: np.ndarray, actions: List):
		
		entry = []
		for idx in range(self._n_agents):
			entry += [' '.join([str(x) for x in obs[idx]]), str(actions[idx])]
		
		return entry


# noinspection PyTypeChecker,DuplicatedCode
class LegibleSingleMADQN(SingleModelMADQN):
	_optimal_models: Dict[str, TrainState]
	_goal_ids: List[str]
	_goal: str
	_beta: float
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float, beta: float,
				 action_space: Space, observation_space: Space, use_gpu: bool, handle_timeout: bool, models_dir: Path, model_names: Dict[str, str], goal: str,
				 dueling_dqn: bool = False, use_ddqn: bool = False, use_vdn: bool = False, use_cnn: bool = False, n_legible_agents: int = 1, cnn_properties: List = None,
				 buffer_data: tuple = (False, '')):
		
		super().__init__(num_agents, action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, action_space, observation_space, use_gpu,
						 dueling_dqn, use_ddqn, use_vdn, use_cnn, handle_timeout, cnn_properties, buffer_data=buffer_data)
		
		self._n_leg_agents = n_legible_agents
		self._goal_ids = list(model_names.keys())
		self._goal = goal
		self._beta = beta
		self._optimal_models = {}
		for goal_id in model_names.keys():
			file_path = models_dir / model_names[goal_id]
			obs_shape = (0,) if not use_cnn else (*observation_space.shape[1:], observation_space.shape[0])
			template = TrainState.create(apply_fn=self._agent_dqn.q_network.apply,
										 params=self._agent_dqn.q_network.init(jax.random.PRNGKey(201), jnp.empty(obs_shape)),
										 tx=optax.adam(learning_rate=0.0))
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
	
	@property
	def beta(self) -> float:
		return self._beta
	
	@property
	def goal(self) -> str:
		return self._goal
	
	@property
	def n_leg_agents(self) -> int:
		return self._n_leg_agents
	
	#####################
	### Class Methods ###
	#####################
	def mse_loss(self, params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray, next_q_value: jnp.ndarray):
		q = jnp.zeros(next_q_value.shape[0])
		for idx in range(self._n_agents):
			if idx < self._n_leg_agents:
				qa = self._agent_dqn.q_network.apply(params, observations[:, idx])
			else:
				qa = self._agent_dqn.q_network.apply(self._optimal_models[self._goal].params, observations[:, idx])
			q += qa[jnp.arange(qa.shape[0]), actions[:, idx].squeeze()]
		q = q.reshape(-1)
		return ((q - next_q_value) ** 2).mean(), q
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_dqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							 next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros(n_obs, dtype=jnp.float32)
		for idx in range(self._n_agents):
			if idx < self._n_leg_agents:
				next_q_value += self._agent_dqn.compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx], target_state_params)
			else:
				next_q_value += self._agent_dqn.compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx],
																	self._optimal_models[self._goal].params)
				# next_q_value += self._agent_dqn.q_network.apply(self._optimal_models[self._goal].params, next_observations[:, idx]).max(axis=1)
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_pred, q_state
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_ddqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							  next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros(n_obs, dtype=jnp.float32)
		for idx in range(self._n_agents):
			if idx < self._n_leg_agents:
				next_q_value += self._agent_dqn.compute_ddqn_targets(dones, next_observations[:, idx], rewards[:, idx].reshape(-1, 1), target_state_params,
																	 q_state.params).squeeze()
			else:
				next_q_value += self._agent_dqn.compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx],
																	self._optimal_models[self._goal].params)
				# next_q_value += self._agent_dqn.q_network.apply(self._optimal_models[self._goal].params, next_observations[:, idx]).max(axis=1)
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_pred, q_state

	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
				  eps_type: str, rng_seed: int, logger: logging.Logger, cnn_shape: Tuple[int], exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1,
				  target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0, greedy_action: bool = True, epoch_logging: bool = False,
				  initial_model_path: str = '', use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = ''):
			
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)
		
		# Setup DQNs for training
		logger.info('Initializing network')
		obs, _ = env.reset()
		self.initialize_network(cnn_shape, logger, obs, optim_learn_rate, rng_seed, initial_model_path)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		avg_episode_len = []
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			episode_start = epoch
			avg_loss = []
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
					for a_idx in range(self._n_agents):
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
						# action = jax.device_get(action)
						episode_q_vals += (float(q_values[int(action)]) / self._n_agents)
						actions += [action]
					actions = np.array(actions)
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				episode_history += [self.get_history_entry(obs, actions)]
				if use_render:
					env.render()
				
				# Obtain the legible rewards
				legible_rewards = np.zeros(self._n_agents)
				n_goals = len(self._optimal_models)
				for a_idx in range(self._n_agents):
					act_q_vals = np.zeros(n_goals)
					action = actions[a_idx]
					for g_idx in range(n_goals):
						if self._agent_dqn.cnn_layer:
							obs_reshape = obs[a_idx].reshape((1, *obs[a_idx].shape))
							q_vals = self._agent_dqn.q_network.apply(self._optimal_models[self._goal_ids[g_idx]].params, obs_reshape)[0]
						else:
							q_vals = self._agent_dqn.q_network.apply(self._optimal_models[self._goal_ids[g_idx]].params, obs[a_idx])
						act_q_vals[g_idx] = np.exp(self._beta * (q_vals[action] - q_vals.max()))
					legible_rewards[a_idx] = act_q_vals[self._goal_ids.index(self._goal)] / act_q_vals.sum()
					episode_rewards += (legible_rewards[a_idx] / self._n_agents)
				
				if terminated:
					finished = np.ones(self._n_agents)
					legible_rewards = legible_rewards / (1 - self.agent_dqn.gamma)
				else:
					finished = np.zeros(self._n_agents)
				
				if use_tracker and epoch_logging:
					performance_tracker.log({
							tracker_panel + "-charts/performance/reward": sum(rewards),
							tracker_panel + "-charts/performance/legible_reward": sum(legible_rewards) / self._n_agents},
							step=(epoch + start_record_epoch))
				
				# store new samples
				if self._use_vdn:
					self.replay_buffer.add(obs, next_obs, actions, legible_rewards, finished[0], infos)
				else:
					for a_idx in range(self._n_agents):
						self.replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], legible_rewards[a_idx], finished[a_idx], infos)
				obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						loss = self.update_model(batch_size, epoch, start_time, tensorboard_frequency, logger)
						if use_tracker and epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss": loss}, step=epoch)
						else:
							avg_loss += [loss]
					
					if epoch % target_freq == 0:
						self._agent_dqn.update_target_model(tau)
				
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
								tracker_panel + "-charts/performance/episodic_length": episode_len,
								tracker_panel + "-charts/performance/avg_episode_length": np.mean(avg_episode_len),
								tracker_panel + "-charts/control/iteration": it,
								tracker_panel + "-charts/control/cycle": cycle,
								tracker_panel + "-charts/control/exploration": eps,
						},
								step=(it + start_record_it))
						if not epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss" : sum(avg_loss) / max(len(avg_loss), 1)},
																	step=(it + start_record_it))
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
	_replay_buffer: ReplayBuffer
	_use_vdn: bool
	_use_ddqn: bool
	_joint_action_converter: Callable
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_converter: Callable, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 action_space: Space, observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_cnn: bool = False,
				 handle_timeout: bool = False, cnn_properties: List[int] = None):
		
		self._num_agents = num_agents
		self._joint_action_converter = act_converter
		self._madqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties, ma_obs=True, n_obs=num_agents)
		self._replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu", handle_timeout_termination=handle_timeout,
										   n_agents=num_agents, smart_add=buffer_data[0], add_method=buffer_data[1])
		
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
	def get_joint_action(self) -> Callable:
		return self._joint_action_converter
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
		
	#####################
	### Class Methods ###
	#####################
	def train_dqn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
				  eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100,
				  use_render: bool = False, cycle: int = 0, use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = ''):
		
		rng_gen = np.random.default_rng(rng_seed)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		self._madqn.init_network_states(rng_seed, obs.reshape((1, *obs.shape)), optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		avg_loss = []
		avg_episode_len = []
		
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
					joint_action = np.array(rng_gen.choice(range(self._madqn.q_network.action_dim)))
				else:
					q_values = self._madqn.q_network.apply(self._madqn.online_state.params, obs.reshape((1, *obs.shape)))[0]
					joint_action = q_values.argmax(axis=-1)
					joint_action = jax.device_get(joint_action)
					episode_q_vals += float(q_values[int(joint_action)])
				actions = self._joint_action_converter(joint_action, self._num_agents)
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				episode_history += [self.get_history_entry(obs, actions)]
				if use_render:
					env.render()
				
				if len(rewards) == 1:
					rewards = np.array([rewards] * self._num_agents)
				
				if terminated or timeout:
					finished = np.ones(self._num_agents)
				else:
					finished = np.zeros(self._num_agents)
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, joint_action, rewards, finished[0], infos)
				episode_rewards += sum(rewards) / self._num_agents
				if use_tracker:
					performance_tracker.log({"charts/performance/reward": sum(rewards)}, step=(epoch + start_record_epoch))
				obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						loss = jax.device_get(self.update_model(batch_size))
						avg_loss.append(loss)
					
					if epoch % target_freq == 0:
						self._madqn.update_target_model(tau)
				
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
								tracker_panel + "-charts/performance/episodic_length": episode_len,
								tracker_panel + "-charts/performance/avg_episode_length": np.mean(avg_episode_len),
								tracker_panel + "-charts/control/iteration": it,
								tracker_panel + "-charts/control/cycle": cycle,
								tracker_panel + "-charts/control/exploration": eps,
						},
								step=(it + start_record_it))
						if not epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss" : sum(avg_loss) / max(len(avg_loss), 1)},
																	step=(it + start_record_it))
					obs, *_ = env.reset()
					done = True
					history += [episode_history]
					episode_rewards = 0
					episode_start = epoch
		
		return history
	
	def update_model(self, batch_size):
		data = self._replay_buffer.sample(batch_size)
		observations = data.observations
		next_observations = data.next_observations
		actions = data.actions
		rewards = data.rewards.sum(axis=1) / self._num_agents
		dones = data.dones
		
		loss = self._madqn.update_online_model(observations, actions, next_observations, rewards, dones)
		return loss
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqn.save_model(filename + '_ctce', model_dir, logger)
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: Tuple) -> None:
		self._agent_dqn.load_model(filename, model_dir, logger, obs_shape)
	
	def get_history_entry(self, obs: np.ndarray, actions: List):
		
		entry = []
		for idx in range(self._num_agents):
			entry += [' '.join([str(x) for x in obs[idx]]), str(actions[idx])]
		
		return entry
	

class MultiEnvSingleMADQN(SingleModelMADQN):
	_n_envs: int
	
	def __init__(self, num_agents: int, num_envs: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 action_space: Space, observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_vdn: bool = False, use_cnn: bool = False,
				 handle_timeout: bool = False, cnn_properties: List = None, buffer_data: tuple = (False, '')):
		
		"""
		Initialize a multi-agent scenario DQN with a single DQN model using multiple simultaneous environments for training

		:param num_agents: number of agents in the environment
		:param num_envs: number of simultaneous environments
		:param action_dim: number of actions of the agent, the DQN is agnostic to the semantic of each action
		:param num_layers: number of layers for the q_network
		:param act_function: activation function for the q_network
		:param layer_sizes: number of neurons in each layer (list must have equal number of entries as the number of layers)
		:param buffer_size: buffer size for the replay buffer
		:param gamma: reward discount factor
		:param observation_space: gym space for the agent observations
		:param use_gpu: flag that controls use of cpu or gpu
		:param handle_timeout: flag that controls handle timeout termination (due to timelimit) separately and treat the task as infinite horizon task.

		:type num_agents: int
		:type num_envs: int
		:type action_dim: int
		:type num_layers: int
		:type buffer_size: int
		:type layer_sizes: list[int]
		:type use_gpu: bool
		:type handle_timeout: bool
		:type gamma: float
		:type act_function: callable
		:type observation_space: gym.Space
		"""
		
		self._n_envs = num_envs
		super().__init__(num_agents, action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, action_space, observation_space, use_gpu,
						 dueling_dqn, use_ddqn, use_vdn, use_cnn, handle_timeout, cnn_properties, n_envs=num_envs, buffer_data=buffer_data)
	
	########################
	### Class Properties ###
	########################
	@property
	def num_envs(self) -> int:
		return self._n_envs
	
	#######################
	### Class Utilities ###
	#######################
	def multi_env_step(self, envs: List[gymnasium.Env], envs_obs: np.ndarray, rng_gen: np.random.Generator, eps: float, greedy_action: bool) -> Tuple:
			
		next_envs_obs = []
		envs_actions = []
		envs_rewards = []
		envs_dones = []
		envs_info = []
		envs_over = [False] * self._n_envs
		explore = rng_gen.random() < eps
		step_q_vals = 0
		
		for env_idx, env in enumerate(envs):
			obs = envs_obs[env_idx]
			if explore:  # Exploration
				actions = np.array(env.action_space.sample())
			
			else:  # Exploitation
				actions = []
				for a_idx in range(self._n_agents):
					# Compute q_values
					if self._agent_dqn.cnn_layer:
						obs_shape = obs[a_idx].shape
						cnn_obs = obs[a_idx].reshape((1, *obs_shape[1:], obs_shape[0]))
						q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, cnn_obs)[0]
					else:
						q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx])
					
					# Get action
					if greedy_action:
						action = q_values.argmax()
					else:
						pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
						pol = pol / pol.sum()
						action = rng_gen.choice(range(env.action_space[0].n), p=pol)
					step_q_vals += float(q_values[int(action)]) / (self._n_agents * self._n_envs)
					actions += [action]
				actions = jnp.array(actions)
			
			next_obs, rewards, terminated, timeout, infos = env.step(actions)
		
			if len(rewards) == 1:
				rewards = np.array([rewards] * self._n_agents)
			
			if terminated:
				finished = np.ones(self._n_agents)
			else:
				finished = np.zeros(self._n_agents)
			
			envs_over[env_idx] = terminated or timeout
			
			next_envs_obs.append(next_obs)
			envs_actions.append(actions)
			envs_rewards.append(rewards)
			envs_dones.append(finished)
			envs_info.append({"timeout": timeout})
			
		next_envs_obs = np.array(next_envs_obs)
		envs_actions = np.array(envs_actions)
		envs_rewards = np.array(envs_rewards)
		envs_dones = np.array(envs_dones)
		envs_info = np.array(envs_info)
		
		step_rewards = 0
		if self.use_vdn:
			self.replay_buffer.add(envs_obs, next_envs_obs, envs_actions, envs_rewards, envs_dones[:, 0], envs_info)
			step_rewards += envs_rewards.sum() / (self._n_agents * self._n_envs)
		else:
			for a_idx in range(self._n_agents):
				self.replay_buffer.add(envs_obs[:, a_idx], next_envs_obs[:, a_idx], envs_actions[:, a_idx], envs_rewards[:, a_idx],
									   envs_dones[:, a_idx], envs_info)
				step_rewards += sum(envs_rewards[:, a_idx])
			step_rewards /= (self._n_agents * self._n_envs)
		
		return next_envs_obs, all(envs_over), step_rewards, step_q_vals
		
	#####################
	### Class Methods ###
	#####################
	@partial(jit, static_argnums=(0,))
	def compute_vdn_dqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							 next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros(n_obs)
		for idx in range(self._n_agents):
			next_q_value += self._agent_dqn.compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx], target_state_params)
		# next_q_value = next_q_value / n_agents
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		return loss_value, q_pred, grads
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_ddqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: jnp.ndarray, actions: jnp.ndarray,
							  next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros((n_obs, 1))
		for idx in range(self._n_agents):
			next_q_value += self._agent_dqn.compute_ddqn_targets(dones, next_observations[:, idx], rewards[:, idx].reshape(-1, 1),
																 target_state_params, q_state.params)
		# next_q_value = next_q_value / n_agents
		
		(loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
		return loss_value, q_pred, grads
	
	def train_dqn(self, envs: List[gymnasium.Env], num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, cnn_shape: Tuple[int], exploration_decay: float = 0.99, warmup: int = 0,
				  train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0, greedy_action: bool = True,
				  epoch_logging: bool = False, all_envs: bool = False, use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = ''):
		
		rng_gen = np.random.default_rng(rng_seed)
		# self._replay_buffer.reseed(rng_seed)
		
		# Setup DQNs for training
		envs_obs = np.array([env.reset()[0] for env in envs])
		if not self._agent_dqn.dqn_initialized:
			logger.info('Initializing network')
			if self._agent_dqn.cnn_layer:
				cnn_obs = envs_obs[0][0].reshape((1, *cnn_shape))
				self._agent_dqn.init_network_states(rng_seed, cnn_obs, optim_learn_rate)
			else:
				self._agent_dqn.init_network_states(rng_seed, envs_obs[0][0], optim_learn_rate)
		
		start_time = time.time()
		sys.stdout.flush()
		start_record_it = cycle * num_iterations
		history = []
		epoch = 0
		avg_episode_len = []
		
		for it in range(num_iterations):
			avg_loss = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			
			if eps_type == 'epoch':
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
			else:
				eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
			
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			episode_start = epoch
			while not done:
			
				next_obs, envs_finished, step_rewards, step_q_vals = self.multi_env_step(envs, envs_obs, rng_gen, eps, greedy_action)
				
				episode_q_vals += step_q_vals
				episode_rewards += step_rewards
				envs_obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						loss = jax.device_get(self.update_model(batch_size, epoch, start_time, tensorboard_frequency, logger, cnn_shape, all_envs))
						avg_loss = [loss]
					
					if epoch % target_freq == 0:
						self._agent_dqn.update_target_model(tau)
				
				epoch += 1
				sys.stdout.flush()
				
				# Check if iteration is over
				if envs_finished:
					episode_len = epoch - episode_start
					avg_episode_len.append(episode_len)
					if use_tracker:
						performance_tracker.log({
								tracker_panel + "-charts/performance/mean_episode_q_vals": 		episode_q_vals / episode_len,
								tracker_panel + "-charts/performance/mean_episode_return": 		episode_rewards / episode_len,
								tracker_panel + "-charts/performance/episodic_length":     		episode_len,
								tracker_panel + "-charts/performance/avg_episode_length":  		np.mean(avg_episode_len),
								tracker_panel + "-charts/performance/iteration":          		it,
								tracker_panel + "-charts/control/cycle":               			cycle,
								tracker_panel + "-charts/control/exploration": 					eps,
								tracker_panel + "-charts/losses/td_loss": 						sum(avg_loss) / max(len(avg_loss), 1)},
								step=(it + start_record_it))
					logger.debug("Episode over:\tLength: %d\tEpsilon: %.5f\tReward: %f" % (episode_len, eps, episode_rewards))
					envs_obs = np.array([env.reset()[0] for env in envs])
					done = True
					episode_rewards = 0
					episode_start = epoch
		
		return history
	
	def update_model(self, batch_size, epoch, start_time, tensorboard_frequency, logger: logging.Logger, cnn_shape: Tuple[int] = None, all_envs: bool = False):
		data = self._replay_buffer.sample(batch_size, all_envs=all_envs)
		observations = data.observations
		next_observations = data.next_observations
		actions = data.actions
		rewards = data.rewards
		dones = data.dones
		n_envs = self._n_envs if all_envs else 1
		# train_info = ('epoch: %d \t' % epoch)
		
		if self._use_vdn:
			loss = 0
			env_grads = []
			for i in range(n_envs):
				if self._agent_dqn.cnn_layer and cnn_shape is not None:
					env_observations = observations[:, i].reshape((*observations[:, i].shape[:2], *cnn_shape))
					env_next_observations = next_observations[:, i].reshape((*next_observations[:, i].shape[:2], *cnn_shape))
				else:
					env_observations = observations[:, i]
					env_next_observations = next_observations[:, i]
					
				if self._use_ddqn:
					env_loss, q_pred, grads = self.compute_vdn_ddqn_loss(self._agent_dqn.online_state, self._agent_dqn.target_params,
																	 env_observations, actions[:, i], env_next_observations, rewards[:, i], dones[:, i])
				else:
					env_loss, q_pred, grads = self.compute_vdn_dqn_loss(self._agent_dqn.online_state, self._agent_dqn.target_params,
																	env_observations, actions[:, i], env_next_observations, rewards[:, i], dones[:, i])
				env_grads += [grads]
				loss += float(env_loss)
			
			for grads in env_grads:
				self.agent_dqn.online_state = self.agent_dqn.online_state.apply_gradients(grads=grads)
			
			loss = loss / n_envs
			
		else:
			if self._agent_dqn.cnn_layer and cnn_shape is not None:
				observations = observations.reshape((len(observations), *cnn_shape))
				next_observations = next_observations.reshape((len(next_observations), *cnn_shape))
			loss = self._agent_dqn.update_online_model(observations, actions, next_observations, rewards, dones)
		
		return loss
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqn.save_model(filename + '_single_model_multi_envs', model_dir, logger)
	
	def get_history_entry(self, obs: np.ndarray, actions: List):
		
		entry = []
		for idx in range(self._n_agents):
			entry += [' '.join([str(x) for x in obs[idx]]), str(actions[idx])]
		
		return entry
