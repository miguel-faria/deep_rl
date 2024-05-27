#! /usr/bin/env python
import sys
import time
import gymnasium
import jax
import numpy as np
import logging

from pathlib import Path
from dl_algos.dqn import DQNetwork, EPS_TYPE
from dl_utilities.buffers import ReplayBuffer
from typing import List, Callable
from datetime import datetime
from gymnasium.spaces import Space


class SingleAgentDQN(object):
	
	_agent_dqn: DQNetwork
	_write_tensorboard: bool
	_replay_buffer: ReplayBuffer

	def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float, action_space: Space,
				 observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_vdn: bool = False, use_cnn: bool = False,
				 handle_timeout: bool = False, use_tensorboard: bool = False, tensorboard_data: List = None, cnn_properties: List[int] = None):
		
		self._write_tensorboard = use_tensorboard
		now = datetime.now()
		if use_tensorboard and tensorboard_data is not None:
			log_name = (tensorboard_data[0] + '/single_model_' + ('vdn_' if use_vdn else '') + now.strftime("%Y%m%d-%H%M%S"))
			if len(tensorboard_data) == 4:
				board_data = [log_name, tensorboard_data[1], tensorboard_data[2], tensorboard_data[3], 'central_train']
			else:
				board_data = [log_name + '_' + tensorboard_data[4], tensorboard_data[1], tensorboard_data[2], tensorboard_data[3], 'central_train']
		else:
			board_data = tensorboard_data
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, use_tensorboard,
									board_data, cnn_properties)
		self._replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu",
										   handle_timeout_termination=handle_timeout)
	
	########################
	### Class Properties ###
	########################
	@property
	def agent_dqn(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
	
	#####################
	### Class Methods ###
	#####################
	def update_dqn_model(self, batch_size: int, epoch: int, start_time: float, always_log: bool, target_freq: int, tau: float,
						 train_freq: int, warmup: int):
		if epoch > warmup:
			if epoch % train_freq == 0:
				data = self._replay_buffer.sample(batch_size)
				observations = data.observations
				next_observations = data.next_observations
				actions = data.actions
				rewards = data.rewards
				dones = data.dones
				
				self._agent_dqn.update_online_model(observations, actions, next_observations, rewards, dones)
			
			if epoch % target_freq == 0:
				self._agent_dqn.update_target_model(tau)
	
	def train(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
			  final_eps: float, eps_type: str, rng_seed: int, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000, train_freq: int = 10,
			  cycle: int = 0, epoch_logging: bool = False):
		
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)
		
		obs, *_ = env.reset()
		self._agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		
		for it in range(num_iterations):
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			qs_count = 0
			episode_start = epoch
			episode_history = []
			print("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				print("Epoch %d" % (epoch + 1))
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				if rng_gen.random() < eps:
					action = np.array([env.action_space.sample()])
				else:
					if self._agent_dqn.cnn_layer:
						q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs.reshape((1, *obs.shape)))[0]
					else:
						q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs)
					action = q_values.argmax(axis=-1)
					# action = jax.device_get(action)
					episode_q_vals += float(q_values[int(action)])
					qs_count += 1
				next_obs, reward, finished, timeout, info = env.step(action)
				episode_rewards += reward
				episode_history += [obs, action]
				if self._write_tensorboard and epoch_logging:
					self._agent_dqn.summary_writer.add_scalar("charts/reward", reward, epoch + start_record_epoch)
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, action, reward, np.array(finished), info)
				obs = next_obs
				
				# update Q-network and target network
				self.update_dqn_model(batch_size, epoch, start_time, epoch_logging, target_freq, tau, train_freq, warmup)
				
				epoch += 1
				sys.stdout.flush()
				if finished:
					obs, _ = env.reset()
					done = True
					history += [episode_history]
					if self._write_tensorboard:
						episode_len = epoch - episode_start
						self._agent_dqn.summary_writer.add_scalar("charts/mean_episode_q_vals", episode_q_vals / qs_count, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/episode_return", episode_rewards, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/mean_episode_return", episode_rewards / episode_len, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
						self._agent_dqn.tensorboard_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
						self._agent_dqn.tensorboard_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/iteration", it, it + start_record_it)
						print("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, epoch - episode_start))
		
		return history
	
	def train_cnn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0,
				  target_freq: int = 1000, train_freq: int = 10, cycle: int = 0, epoch_logging: bool = False):
		
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)

		env.reset()
		obs = env.render()
		self._agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		
		for it in range(num_iterations):
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			qs_count = 0
			episode_start = epoch
			episode_history = []
			logger.info("Iteration %d out of %d" % (it + 1, num_iterations))
			while not done:
				logger.debug("Epoch %d" % (epoch + 1))
				
				# interact with environment
				if eps_type == 'epoch':
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
				else:
					eps = DQNetwork.eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				
				if rng_gen.random() < eps:
					action = np.array(env.action_space.sample())
				else:
					q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs)
					action = q_values.argmax(axis=-1)
					# action = jax.device_get(action)
					episode_q_vals += float(q_values[int(action)])
					qs_count += 1
				_, reward, finished, timeout, info, *_ = env.step(action)
				next_obs = env.render()
				episode_rewards += reward
				episode_history += [obs, action]
				if self._write_tensorboard and epoch_logging:
					self._agent_dqn.summary_writer.add_scalar("charts/reward", reward, epoch + start_record_epoch)
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, action, reward, finished, info)
				obs = next_obs
				
				# update Q-network and target network
				self.update_dqn_model(batch_size, epoch, start_time, epoch_logging, target_freq, tau, train_freq, warmup)
				
				epoch += 1
				sys.stdout.flush()
				if finished:
					env.reset()
					obs = env.render()
					done = True
					history += [episode_history]
					if self._write_tensorboard:
						episode_len = epoch - episode_start
						self._agent_dqn.summary_writer.add_scalar("charts/mean_episode_q_vals", episode_q_vals / qs_count, epoch + start_record_epoch)
						self._agent_dqn.summary_writer.add_scalar("charts/episode_return", episode_rewards, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/mean_episode_return", episode_rewards / episode_len, it + start_record_it)
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", episode_len, it + start_record_it)
						self._agent_dqn.tensorboard_writer.add_scalar("charts/epsilon", eps, epoch)
						logger.debug("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, epoch - episode_start))
		
		return history
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqn.save_model(filename, model_dir, logger)
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: tuple) -> None:
		self._agent_dqn.load_model(filename, model_dir, logger, obs_shape)