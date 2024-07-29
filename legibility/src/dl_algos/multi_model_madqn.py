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
from typing import List, Dict, Callable, Optional
from functools import partial
from jax import jit
from wandb.wandb_run import Run


COEF = 1.0


class MultiAgentDQN(object):
	
	_num_agents: int
	_agent_ids: List[str]
	_agent_dqns: Dict[str, DQNetwork]
	_replay_buffer: ReplayBuffer
	_use_vdn: bool
	_use_ddqn: bool
	
	def __init__(self, num_agents: int, agent_ids: List[str], action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 action_space: Space, observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_vdn: bool = False, use_cnn: bool = False,
				 handle_timeout: bool = False, cnn_properties: List[int] = None):
		
		"""
		Initialize a multi-agent scenario DQN with decentralized training and execution
		
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
        the summary writer (default is None)
        
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
		
		self._num_agents = num_agents
		self._agent_ids = agent_ids
		self._use_vdn = use_vdn
		self._use_ddqn = use_ddqn
		self._agent_dqns = {}
		self._replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu",
										   handle_timeout_termination=handle_timeout, n_agents=num_agents)
		for agent_id in agent_ids:
			self._agent_dqns[agent_id] = DQNetwork(action_dim, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties)
	
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._num_agents
	
	@property
	def agent_ids(self) -> List[str]:
		return self._agent_ids
	
	@property
	def agent_dqns(self) -> Dict[str, DQNetwork]:
		return self._agent_dqns
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
	
	@property
	def use_vdn(self) -> bool:
		return self._use_vdn
	
	#####################
	### Class Methods ###
	#####################
	def train_dqns(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
				   eps_type: str, rng_seed: int, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100, tensorboard_frequency: int = 1,
				   use_render: bool = False, cycle: int = 0, use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = ''):
		
		rng_gen = np.random.default_rng(rng_seed)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		env.action_space.seed(rng_seed)
		env.observation_space.seed(rng_seed)
		
		for a_id in self._agent_ids:
			agent_dqn = self._agent_dqns[a_id]
			agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		sys.stdout.flush()
		
		for it in range(num_iterations):
			if use_render:
				env.render()
			done = False
			episode_rewards = [0] * self._num_agents
			episode_start = epoch
			print("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				
				# interact with environment
				if eps_type == 'linear':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				if rng_gen.random() < eps:
					actions = np.array(env.action_space.sample())
				else:
					actions = []
					for a_id in self._agent_ids:
						agent_dqn = self._agent_dqns[a_id]
						a_idx = self._agent_ids.index(a_id)
						q_values = agent_dqn.q_network.apply(agent_dqn.online_state.params, obs[a_idx])
						action = q_values.argmax(axis=-1)
						action = jax.device_get(action)
						if use_tracker and epoch % tensorboard_frequency == 0:
							performance_tracker.log({
									tracker_panel + '_' + a_id + "-charts/performance/episodic_q_vals": float(q_values[int(action)]),
							},
									step=(epoch + start_record_epoch))
						actions += [action]
					actions = np.array(actions)
				next_obs, rewards, terminated, timeout, infos = env.step(actions)
				if use_render:
					env.render()
				
				# update accumulated rewards
				if len(rewards) == 1:
					rewards = np.array([rewards] * self._num_agents)
				
				if terminated:
					finished = np.ones(self._num_agents)
				else:
					finished = np.zeros(self._num_agents)
				
				# store new samples and update episode rewards
				self._replay_buffer.add(obs, next_obs, actions, rewards, finished, [infos])
				for a_idx in range(self._num_agents):
					a_id = self._agent_ids[a_idx]
					episode_rewards[a_idx] += rewards[a_idx]
					if use_tracker:
						performance_tracker.log({tracker_panel + '_' + a_id + "-charts/performance/reward": rewards[a_idx]}, step=(epoch + start_record_epoch))
				obs = next_obs
				
				# update Q-network and target network
				self.update_dqn_models(batch_size, epoch, target_freq, tau, tensorboard_frequency, train_freq, warmup, use_tracker, performance_tracker, tracker_panel)
				
				epoch += 1
				sys.stdout.flush()
				
				# Check if iteration is over
				if terminated or timeout:
					done = True
					obs, *_ = env.reset()
					for a_idx in range(self._num_agents):
						if use_tracker:
							a_id = self._agent_ids[a_idx]
							performance_tracker.log({
									tracker_panel + '_' + a_id + "-charts/performance/episodic_return": episode_rewards[a_idx],
									tracker_panel + '_' + a_id + "-charts/performance/episodic_length": epoch - episode_start,
									tracker_panel + '_' + a_id + "-charts/control/epsilon": eps,
									tracker_panel + '_' + a_id + "-charts/control/iteration": it},
									step=(it + start_record_it))
						print("Agent %s episode over:\tReward: %f\tLength: %d" % (self._agent_ids[a_idx], episode_rewards[a_idx], epoch - episode_start))
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_dqn_loss(self, q_state: List[TrainState], target_state_params: List[flax.core.FrozenDict], observations: jnp.ndarray, actions: jnp.ndarray,
							 next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros(n_obs)
		q_vals = jnp.zeros(n_obs)
		for idx in range(self._num_agents):
			next_q_value += self._agent_dqns[self._agent_ids[idx]].compute_dqn_targets(dones, next_observations[:, idx], rewards[:, idx].reshape(-1, 1),
																					   target_state_params[idx])
			qa = self._agent_dqns[self._agent_ids[idx]].q_network.apply(q_state[idx].params, observations[:, idx])
			q_vals += qa[np.arange(qa.shape[0]), actions[:, idx].squeeze()]
		q_vals = q_vals.reshape(-1, 1)
		# next_q_value = next_q_value / n_agents
		
		new_q_states = []
		loss_value, grads = jax.value_and_grad(optax.l2_loss)(q_vals, next_q_value)
		for idx in range(len(q_state)):
			new_q_states.append(q_state[idx].apply_gradients(grads=grads))
		return loss_value, q_vals, new_q_states
	
	@partial(jit, static_argnums=(0,))
	def compute_vdn_ddqn_loss(self, q_state: List[TrainState], target_state_params: List[flax.core.FrozenDict], observations: jnp.ndarray, actions: jnp.ndarray,
							 next_observations: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray):
		n_obs = len(observations)
		next_q_value = jnp.zeros(n_obs)
		q_vals = jnp.zeros(n_obs)
		for idx in range(self._num_agents):
			next_q_value += self._agent_dqns[self._agent_ids[idx]].compute_ddqn_targets(dones, next_observations[:, idx], rewards[:, idx].reshape(-1, 1),
																						target_state_params[idx], q_state[idx].params)
			qa = self._agent_dqns[self._agent_ids[idx]].q_network.apply(q_state[idx].params, observations[:, idx])
			q_vals += qa[np.arange(qa.shape[0]), actions[:, idx].squeeze()]
		q_vals = q_vals.reshape(-1, 1)
		# next_q_value = next_q_value / n_agents
		
		new_q_states = []
		loss_value, grads = jax.value_and_grad(optax.l2_loss)(q_vals, next_q_value)
		for idx in range(len(q_state)):
			new_q_states.append(q_state[idx].apply_gradients(grads=grads))
		return loss_value, q_vals, new_q_states
		
	def update_dqn_models(self, batch_size: int, epoch: int, target_freq: int, tau: float, tensorboard_frequency: int, train_freq: int, warmup: int,
						  use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = ''):
		if epoch >= warmup:
			if epoch % train_freq == 0:
				data = self._replay_buffer.sample(batch_size)
				observations = data.observations
				next_observations = data.next_observations
				actions = data.actions
				rewards = data.rewards
				dones = data.dones
				
				if self._use_vdn:
					q_states = [self._agent_dqns[a_id].online_state for a_id in self._agent_ids]
					target_params = [self._agent_dqns[a_id].target_params for a_id in self._agent_ids]
					if self._use_ddqn:
						loss, q_pred, q_states = self.compute_vdn_ddqn_loss(q_states, target_params, observations, actions, next_observations, rewards, dones)
					else:
						loss, q_pred, q_states = self.compute_vdn_dqn_loss(q_states, target_params, observations, actions, next_observations, rewards, dones)
					
					for a_idx in range(self._num_agents):
						agent_dqn = self._agent_dqns[self._agent_ids[a_idx]]
						agent_dqn.online_state = q_states[a_idx]
						
						#  update tensorboard
						if use_tracker and epoch % tensorboard_frequency == 0:
							performance_tracker.log({
									tracker_panel + '_' + self._agent_ids[a_idx] + "-charts/losses/td_loss": jax.device_get(loss),
									tracker_panel + '_' + self._agent_ids[a_idx] + "-losses/avg_q_values": jax.device_get(q_pred).mean(),
							}, step=epoch)
					
				else:
					losses = []
					for a_idx in range(self._num_agents):
						a_id = self._agent_ids[a_idx]
						loss = self._agent_dqns[a_id].update_online_model(observations[a_idx], actions[a_idx], next_observations[a_idx], rewards[a_idx], dones[a_idx])
						losses += [loss]
					
			if epoch % target_freq == 0:
				for a_id in self._agent_ids:
					self._agent_dqns[a_id].update_target_model(tau)
	
	def save_models(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		for agent_id in self._agent_ids:
			self._agent_dqns[agent_id].save_model(filename, model_dir, logger)
	
	def save_model(self, filename: str, agent_id: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqns[agent_id].save_model(filename, model_dir, logger)
	
	def load_models(self, filename_prefix: str, model_dir: Path, logger: logging.Logger, obs_shape: tuple) -> None:
		for agent_id in self._agent_ids:
			self._agent_dqns[agent_id].load_model(filename_prefix + '_' + agent_id + '.model', model_dir, logger, obs_shape)
	
	def load_model(self, filename: str, agent_id: str, model_dir: Path, logger: logging.Logger, obs_shape: tuple) -> None:
		self._agent_dqns[agent_id].load_model(filename, model_dir, logger, obs_shape)
