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
from typing import List, Callable, Optional
from datetime import datetime
from gymnasium.spaces import Space
from wandb.wandb_run import Run


class SingleAgentDQN(object):
	
	_agent_dqn: DQNetwork
	_replay_buffer: ReplayBuffer

	def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float, action_space: Space,
				 observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False, use_cnn: bool = False, handle_timeout: bool = False,
				 cnn_properties: List[int] = None):
		
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, gamma, dueling_dqn, use_ddqn, use_cnn, cnn_properties)
		self._replay_buffer = ReplayBuffer(buffer_size, observation_space, action_space, "cuda" if use_gpu else "cpu", handle_timeout_termination=handle_timeout)
	
	########################
	### Class Properties ###
	########################
	@property
	def agent_dqn(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def replay_buffer(self) -> ReplayBuffer:
		return self._replay_buffer
	
	#####################
	### Class Methods ###
	#####################
	def train(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
			  final_eps: float, eps_type: str, rng_seed: int, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 1000, train_freq: int = 10,
			  cycle: int = 0, epoch_logging: bool = False, use_tracker: bool = False, performance_tracker: Optional[Run] = None, tracker_panel: str = ''):
		
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)
		
		obs, *_ = env.reset()
		self._agent_dqn.init_network_states(rng_seed, obs, optim_learn_rate)
		
		start_time = time.time()
		epoch = 0
		start_record_it = cycle * num_iterations
		start_record_epoch = cycle * max_timesteps
		history = []
		avg_episode_len = []
		
		for it in range(num_iterations):
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			qs_count = 0
			episode_start = epoch
			episode_history = []
			avg_loss = []
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
				if use_tracker and epoch_logging:
					performance_tracker.log({tracker_panel + "-charts/performance/reward": reward}, step=(epoch + start_record_epoch))
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, action, reward, np.array(finished), info)
				obs = next_obs
				
				# update Q-network and target network
				if epoch > warmup:
					if epoch % train_freq == 0:
						data = self._replay_buffer.sample(batch_size)
						observations = data.observations
						next_observations = data.next_observations
						actions = data.actions
						rewards = data.rewards
						dones = data.dones
						
						loss = jax.device_get(self._agent_dqn.update_online_model(observations, actions, next_observations, rewards, dones))
						if use_tracker and epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss": loss}, step=epoch)
						else:
							avg_loss += [loss]
					
					if epoch % target_freq == 0:
						self._agent_dqn.update_target_model(tau)
				
				epoch += 1
				sys.stdout.flush()
				if finished:
					obs, _ = env.reset()
					done = True
					history += [episode_history]
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
						print("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, epoch - episode_start))
		
		return history
	
	def train_cnn(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				  final_eps: float, eps_type: str, rng_seed: int, logger: logging.Logger, exploration_decay: float = 0.99, warmup: int = 0,
				  target_freq: int = 1000, train_freq: int = 10, cycle: int = 0, epoch_logging: bool = False, use_tracker: bool = False, performance_tracker: Optional[Run] = None,
				  tracker_panel: str = ''):
		
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
		avg_episode_len = []
		
		for it in range(num_iterations):
			done = False
			episode_rewards = 0
			episode_q_vals = 0
			qs_count = 0
			episode_start = epoch
			episode_history = []
			avg_loss = []
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
				if use_tracker and epoch_logging:
					performance_tracker.log({tracker_panel + "-charts/performance/reward": reward}, step=(epoch + start_record_epoch))
				
				# store new samples
				self._replay_buffer.add(obs, next_obs, action, reward, finished, info)
				obs = next_obs
				
				# update Q-network and target network
				if epoch > warmup:
					if epoch % train_freq == 0:
						data = self._replay_buffer.sample(batch_size)
						observations = data.observations
						next_observations = data.next_observations
						actions = data.actions
						rewards = data.rewards
						dones = data.dones
						
						loss = jax.device_get(self._agent_dqn.update_online_model(observations, actions, next_observations, rewards, dones))
						if use_tracker and epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss": loss}, step=epoch)
						else:
							avg_loss += [loss]
					
					if epoch % target_freq == 0:
						self._agent_dqn.update_target_model(tau)
				
				epoch += 1
				sys.stdout.flush()
				if finished:
					env.reset()
					obs = env.render()
					done = True
					history += [episode_history]
					episode_len = epoch - episode_start
					avg_episode_len += [episode_len]
					if use_tracker:
						performance_tracker.log({
								tracker_panel + "-charts/performance/mean_episode_q_vals": episode_q_vals / episode_len,
								tracker_panel + "-charts/performance/mean_episode_return": episode_rewards / episode_len,
								tracker_panel + "-charts/performance/episodic_length":     episode_len,
								tracker_panel + "-charts/performance/avg_episode_length":  np.mean(avg_episode_len),
								tracker_panel + "-charts/control/iteration":               it,
								tracker_panel + "-charts/control/cycle":                   cycle,
								tracker_panel + "-charts/control/cycle": cycle,
						},
								step=(it + start_record_it))
						if not epoch_logging:
							performance_tracker.log({tracker_panel + "-charts/losses/td_loss" : sum(avg_loss) / max(len(avg_loss), 1)},
													step=(it + start_record_it))
					logger.debug("Episode over:\tReward: %f\tLength: %d" % (episode_rewards, episode_len))
		
		return history
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		self._agent_dqn.save_model(filename, model_dir, logger)
	
	def load_model(self, filename: str, model_dir: Path, logger: logging.Logger, obs_shape: tuple) -> None:
		self._agent_dqn.load_model(filename, model_dir, logger, obs_shape)