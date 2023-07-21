#! /usr/bin/env python
import sys

import math
import random
import time
import flax
import flax.linen as nn
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import optax
import json

from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Discrete, Space
from pathlib import Path
from termcolor import colored
from dl_algos.dqn import DQNetwork, QNetwork, EPS_TYPE
from typing import List, Dict, Callable, Set
from datetime import datetime


COEF = 1.0


class MultiAgentDQN(object):
	
	_num_agents: int
	_agent_ids: List[str]
	_agent_dqns: Dict[str, DQNetwork]
	_write_tensorboard: bool
	
	def __init__(self, num_agents: int, agent_ids: List[str], action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int],
				 buffer_size: int, gamma: float, observation_space: Space, use_gpu: bool, dueling_dqn: bool = False, use_ddqn: bool = False,
				 handle_timeout: bool = False, use_tensorboard: bool = False, tensorboard_data: List = None):
		
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
		self._agent_ids = agent_ids
		self._write_tensorboard = use_tensorboard
		
		self._agent_dqns = {}
		self._record_obs = []
		for agent_id in agent_ids:
			now = datetime.now()
			if tensorboard_data is not None:
				board_data = [tensorboard_data[0] + '/' + agent_id + '_' + now.strftime("%Y%m%d-%H%M%S"), tensorboard_data[1], tensorboard_data[2],
							  tensorboard_data[3], agent_id]
			else:
				board_data = tensorboard_data
			self._agent_dqns[agent_id] = DQNetwork(action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, observation_space, use_gpu,
												   dueling_dqn, use_ddqn, handle_timeout, use_tensorboard, board_data)
			self._record_obs += [list()]
	
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
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@num_agents.setter
	def num_agents(self, num_agents: int):
		self._num_agents = num_agents
	
	@agent_ids.setter
	def agent_ids(self, agent_ids: List[str]):
		self._agent_ids = agent_ids
		
	@agent_dqns.setter
	def agent_dqns(self, agent_dqns: Dict[str, DQNetwork]):
		self._agent_dqns = agent_dqns
		
	@write_tensorboard.setter
	def write_tensorboard(self, write_tensorboard: bool):
		self._write_tensorboard = write_tensorboard
	
	#####################
	### Class Methods ###
	#####################
	def train_dqns(self, env: gymnasium.Env, num_iterations: int, max_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
				   final_eps: float, eps_type: str, rng_seed: int, log_filename: str, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1,
				   target_freq: int = 100, tensorboard_frequency: int = 1, use_render: bool = False, cycle: int = 0):
		
		def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, iteration: int, max_iterations: int):
			
			if update_type == 1:
				return max(((final_eps - init_eps) / (max_iterations * decay_rate)) * iteration + init_eps, end_eps)
			elif update_type == 2:
				return max(decay_rate ** iteration * init_eps, end_eps)
			elif update_type == 3:
				return max((1 / (1 + decay_rate * iteration)) * init_eps, end_eps)
			elif update_type == 4:
				return max((decay_rate * math.sqrt(iteration)) * init_eps, end_eps)
			else:
				print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
				return max((1 / (1 + decay_rate * iteration)) * init_eps, end_eps)
		
		random.seed(rng_seed)
		np.random.seed(rng_seed)
		rng_gen = np.random.default_rng(rng_seed)
		key = jax.random.PRNGKey(rng_seed)
		key, q_key = jax.random.split(key, 2)
		
		# Setup DQNs for training
		obs, *_ = env.reset()
		env.action_space.seed(rng_seed)
		env.observation_space.seed(rng_seed)
		
		for a_id in self._agent_ids:
			agent_dqn = self._agent_dqns[a_id]
			a_idx = self._agent_ids.index(a_id)
			if agent_dqn.online_state is None:
				agent_dqn.online_state = TrainState.create(
					apply_fn=agent_dqn.q_network.apply,
					params=agent_dqn.q_network.init(q_key, obs[a_idx]),
					tx=optax.adam(learning_rate=optim_learn_rate),
				)
			
			if agent_dqn.target_params is None:
				agent_dqn.target_params = agent_dqn.q_network.init(q_key, obs[a_idx])
			
			agent_dqn.q_network.apply = jax.jit(agent_dqn.q_network.apply)
			agent_dqn.target_params = optax.incremental_update(agent_dqn.online_state.params, agent_dqn.target_params, 1.0)
		
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
				print("Epoch %d" % (epoch + 1))
				
				# interact with environment
				if eps_type == 'linear':
					eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, it, num_iterations)
				else:
					eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, max_timesteps)
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
						if self._agent_dqns[a_id].use_summary and epoch % tensorboard_frequency == 0:
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/episodic_q_vals", float(q_values[int(action)]), epoch + start_record_epoch)
						actions += [action]
					actions = np.array(actions)
				next_obs, rewards, finished, infos, *_ = env.step(actions)
				if use_render:
					env.render()
				
				# update accumulated rewards
				if len(rewards) == 1:
					rewards = [rewards] * self._num_agents
				
				if len(finished) == 1:
					finished = [finished] * self._num_agents
				
				# store new samples and update episode rewards
				for a_idx in range(self._num_agents):
					a_id = self._agent_ids[a_idx]
					self._agent_dqns[a_id].replay_buffer.add(obs[a_idx], next_obs[a_idx], actions[a_idx], rewards[a_idx], finished[a_idx], infos)	# store sample
					episode_rewards[a_idx] += rewards[a_idx]
					if self._agent_dqns[a_id].use_summary:
						self._agent_dqns[a_id].summary_writer.add_scalar("charts/reward", rewards[a_idx], epoch + start_record_epoch)
				obs = next_obs
				
				# update Q-network and target network
				if epoch >= warmup:
					if epoch % train_freq == 0:
						self.update_models(batch_size, epoch, start_time, tensorboard_frequency)
					
					if epoch % target_freq == 0:
						for a_id in self._agent_ids:
							self._agent_dqns[a_id].update_target_model(tau)
				
				epoch += 1
				sys.stdout.flush()
				if all(finished):
					done = True
					obs, *_ = env.reset()
					if self._write_tensorboard:
						for a_idx in range(self._num_agents):
							a_id = self._agent_ids[a_idx]
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/episodic_return", episode_rewards[a_idx], it + start_record_it)
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, it + start_record_it)
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/epsilon", eps, it + start_record_it)
							print("Agent %s episode over:\tReward: %f\tLength: %d" % (a_id, episode_rewards[a_idx], epoch - episode_start))
	
	def update_models(self, batch_size: int, epoch: int, start_time: float, tensorboard_frequency: int):
		train_info = ('epoch: %d \t' % epoch)
		losses = []
		for a_id in self._agent_ids:
			loss = self._agent_dqns[a_id].update_online_model(batch_size, epoch, start_time, tensorboard_frequency)
			losses += [loss]
			train_info += ('agent %s: loss: %.7f\t' % (a_id, loss))
			
		# print('Train losses: ' + ','.join([str(x) for x in losses]))
		# print('Train Info: ' + train_info)
		return sum(losses) / self._num_agents
	
	def save_models(self, filename: str, model_dir: Path) -> None:
		for agent_id in self._agent_ids:
			file_path = model_dir / (filename + '_' + agent_id + '.model')
			with open(file_path, "wb") as f:
				f.write(flax.serialization.to_bytes(self._agent_dqns[agent_id].online_state))
		print("Model states saved to files")
	
	def save_model(self, filename: str, agent_id: str, model_dir: Path) -> None:
		model_path = model_dir / (filename + '_' + agent_id + '.model')
		with open(model_path, "wb") as f:
			f.write(flax.serialization.to_bytes(self._agent_dqns[agent_id].online_state))
		print("Model state saved to file: " + str(model_path))
	
	def load_models(self, filename_prefix: str, model_dir: Path) -> None:
		for agent_id in self._agent_ids:
			model_path = model_dir / (filename_prefix + '_' + agent_id + '.model')
			template = TrainState.create(apply_fn=self._agent_dqns[agent_id].q_network.apply,
										 params=self._agent_dqns[agent_id].q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7))),
										 tx=optax.adam(learning_rate=0.0001))
			with open(model_path, "rb") as f:
				self._agent_dqns[agent_id].online_state = flax.serialization.from_bytes(template, f.read())
			print("Loaded model states from files")
	
	def load_model(self, filename: str, agent_id: str, model_dir: Path) -> None:
		file_path = model_dir / filename
		template = TrainState.create(apply_fn=self._agent_dqns[agent_id].q_network.apply,
									 params=self._agent_dqns[agent_id].q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7))),
									 tx=optax.adam(learning_rate=0.0001))
		with open(file_path, "rb") as f:
			self._agent_dqns[agent_id].online_state = flax.serialization.from_bytes(template, f.read())
		print("Loaded model state from file: " + str(file_path))
	
	def get_policies(self) -> List[np.ndarray]:
		
		policies = []
		for a_id in self._agent_ids:
			a_pol = []
			a_idx = self._agent_ids.index(a_id)
			for rec_obs in self._record_obs[a_idx]:
				obs = np.array([float(x) for x in rec_obs.split(' ')])
				q_values = self._agent_dqns[a_id].q_network.apply(self._agent_dqns[a_id].online_state.params, obs)
				max_q = q_values.max(axis=-1)
				tmp_pol = np.isclose(q_values, max_q, rtol=1e-10, atol=1e-10).astype(int)
				a_pol += [tmp_pol / sum(tmp_pol)]
			policies += [np.array(a_pol)]
			
		return policies
	
	def get_policy(self, a_idx: int) -> np.ndarray:
		
		pol = []
		a_id = self._agent_ids[a_idx]
		for rec_obs in self._record_obs[a_idx]:
			obs = np.array([float(x) for x in rec_obs.split(' ')])
			q_values = self._agent_dqns[a_id].q_network.apply(self._agent_dqns[a_id].online_state.params, obs)
			max_q = q_values.max(axis=-1)
			tmp_pol = np.isclose(q_values, max_q, rtol=1e-10, atol=1e-10).astype(int)
			pol += [tmp_pol / sum(tmp_pol)]
		
		return pol


class CentralizedTrainingMADQN(object):
	
	_num_agents: int
	_agent_dqn: DQNetwork
	_write_tensorboard: bool
	_record_obs: Set[str]
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 observation_space: Space, use_gpu: bool, handle_timeout: bool, use_tensorboard: bool = False, tensorboard_data: List = None):
		
		"""
		Initialize a multi-agent scenario DQN with centralized training and decentralized execution
		
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
		self._record_obs = []
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, observation_space, use_gpu, handle_timeout,
									use_tensorboard, tensorboard_data)
		
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._num_agents
	
	@property
	def agent_dqns(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@num_agents.setter
	def num_agents(self, num_agents: int):
		self._num_agents = num_agents
	
	@agent_dqns.setter
	def agent_dqns(self, agent_dqn: DQNetwork):
		self._agent_dqn = agent_dqn
	
	@write_tensorboard.setter
	def write_tensorboard(self, write_tensorboard: bool):
		self._write_tensorboard = write_tensorboard
	
	#####################
	### Class Methods ###
	#####################
	def train_dqn(self, env: gymnasium.Env, total_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
				  eps_type: str, rng_seed: int, log_filename: str, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1, target_freq: int = 100,
				  tensorboard_frequency: int = 1):
		
		def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, epoch: int, max_timesteps: int):
			
			if update_type == 1:
				return max(((final_eps - init_eps) / (max_timesteps * decay_rate)) * COEF * epoch + init_eps, end_eps)
			elif update_type == 2:
				return max(decay_rate ** epoch * init_eps, end_eps)
			elif update_type == 3:
				return max((1 / (1 + decay_rate * epoch)) * init_eps, end_eps)
			elif update_type == 4:
				return max((decay_rate * math.sqrt(epoch)) * init_eps, end_eps)
			else:
				print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
				return max((1 / (1 + decay_rate * epoch)) * init_eps, end_eps)
		
		random.seed(rng_seed)
		np.random.seed(rng_seed)
		key = jax.random.PRNGKey(rng_seed)
		key, q_key = jax.random.split(key, 2)
		# train_log_file = Path(__file__).parent.absolute().parent.absolute().parent.absolute() / 'logs' / log_filename
		
		# Setup DQNs for training
		obs, _ = env.reset()
		
		if self._agent_dqn.online_state is None:
			self._agent_dqn.online_state = TrainState.create(
				apply_fn=self._agent_dqn.q_network.apply,
				params=self._agent_dqn.q_network.init(q_key, obs[0]),
				tx=optax.adam(learning_rate=optim_learn_rate),
			)
		
		if self._agent_dqn.target_params is None:
			self._agent_dqn.target_params = self._agent_dqn.q_network.init(q_key, obs[0])
		
		self._agent_dqn.q_network.apply = jax.jit(self._agent_dqn.q_network.apply)
		self._agent_dqn.target_params = optax.incremental_update(self._agent_dqn.online_state.params, self._agent_dqn.target_params, 1.0)
		
		start_time = time.time()
		to_convergence = (total_timesteps < 0)
		done = False
		epoch = 0
		loss = math.inf
		sys.stdout.flush()
		episode_rewards = 0
		episode_start = epoch
		
		while not done:
			print("Epoch %d out of %d" % (epoch + 1, total_timesteps))
			
			# interact with environment
			for a_idx in range(self._num_agents):
				self._record_obs.add(' '.join([str(x) for x in obs[a_idx]]))
			eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, total_timesteps)
			print('Epsilon: %f' % eps)
			if random.random() < eps:
				actions = np.array(env.action_space.sample())
			else:
				actions = []
				for a_idx in range(self._num_agents):
					q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx])
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
			next_obs, rewards, dones, infos, _ = env.step(actions)
			
			if len(rewards) == 1:
				rewards = [rewards] * self._num_agents
			for agent_idx in range(self._num_agents):
				episode_rewards += (rewards[agent_idx] / 2)
			
			if len(dones) == 1:
				dones = [dones] * self._num_agents
			
			if self._write_tensorboard and all(dones):
				self._agent_dqn.summary_writer.add_scalar("charts/episodic_return", episode_rewards, epoch)
				self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", epoch - episode_start, epoch)
				self._agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, epoch)
			
			# store new samples
			real_next_obs = list(next_obs).copy()
			for a_idx in range(self._num_agents):
				# for idx, d in enumerate(dones):
				# 	if d:
				# 		real_next_obs[a_idx][idx] = infos[idx]["terminal_observation"]
				self._agent_dqn.replay_buffer.add(obs[a_idx], real_next_obs[a_idx], actions[a_idx], rewards[a_idx], dones[a_idx], infos)
			obs = next_obs
			
			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					loss = self.update_model(batch_size, epoch, start_time, tensorboard_frequency)
				
				if epoch % target_freq == 0:
					self._agent_dqn.target_params = optax.incremental_update(self._agent_dqn.online_state.params, self._agent_dqn.target_params, tau)
			
			if (to_convergence and abs(loss) < 1e-9) or (not to_convergence and epoch >= total_timesteps):
				done = True
			
			epoch += 1
			sys.stdout.flush()
			if np.all(dones):
				obs, _ = env.reset()
				episode_rewards = 0
				episode_start = epoch
	
	def update_model(self, batch_size, epoch, start_time, tensorboard_frequency):
		train_info = ('epoch: %d \t' % epoch)
		losses = []
		for a_idx in range(self._num_agents):
			device = self._agent_dqn.replay_buffer.device
			data = self._agent_dqn.replay_buffer.sample(batch_size)
			
			# perform a gradient-descent step
			if device == 'cpu':
				loss, old_val, self._agent_dqn.online_state = self._agent_dqn.compute_loss(
					self._agent_dqn.online_state,
					data.observations.numpy(),
					data.actions.numpy(),
					data.next_observations.numpy(),
					data.rewards.flatten().numpy(),
					data.dones.flatten().numpy()
				)
			else:
				loss, old_val, self._agent_dqn.online_state = self._agent_dqn.compute_loss(
					self._agent_dqn.online_state,
					data.observations.cpu().numpy(),
					data.actions.cpu().numpy(),
					data.next_observations.cpu().numpy(),
					data.rewards.flatten().cpu().numpy(),
					data.dones.flatten().cpu().numpy()
				)
			
			losses += [loss]
			train_info += ('agent loss: %.7f\t' % loss)
			
			# write log data for tensorboard
			if self._write_tensorboard and epoch % tensorboard_frequency == 0:
				self._agent_dqn.summary_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
				self._agent_dqn.summary_writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), epoch)
				print("SPS:", int(epoch / (time.time() - start_time)))
				self._agent_dqn.summary_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
		print('Train losses: ' + ','.join([str(x) for x in losses]))
		print('Train Info: ' + train_info)
		return sum(losses) / self._num_agents
	
	def save_model(self, filename: str, model_dir: Path) -> None:
		model_path = model_dir / (filename + '_centralized.model')
		obs_path = model_dir / (filename + '_centralized.json')
		with open(model_path, "wb") as mf:
			mf.write(flax.serialization.to_bytes(self._agent_dqn.online_state))
		with open(obs_path, "w") as of:
			of.write(json.dumps(list(self._record_obs)))
		print("Model state saved to file: " + str(model_path))
	
	def load_model(self, filename: str, model_dir: Path) -> None:
		file_path = model_dir / filename
		template = TrainState.create(apply_fn=self._agent_dqn.q_network.apply,
									 params=self._agent_dqn.q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7))),
									 tx=optax.adam(learning_rate=0.0001))
		with open(file_path, "rb") as f:
			self._agent_dqn.online_state = flax.serialization.from_bytes(template, f.read())
		print("Loaded model state from file: " + str(file_path))
		
	def load_obs(self, filename: str, model_dir: Path) -> None:
		file_path = model_dir / filename
		with open(file_path, "r") as f:
			self._record_obs = set(json.load(f))
		print("Loaded model state from file: " + str(file_path))
	
	def get_policy(self) -> np.ndarray:
		
		pol = []
		for rec_obs in self._record_obs:
			obs = np.array([float(x) for x in rec_obs.split(' ')])
			q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs)
			max_q = q_values.max(axis=-1)
			tmp_pol = np.isclose(q_values, max_q, rtol=1e-10, atol=1e-10).astype(int)
			pol += [tmp_pol / sum(tmp_pol)]
		
		return pol


class SingleControlMADQN(object):
	
	_num_agents: int
	_agent_list: List[str]
	_agent_dqn: DQNetwork
	_write_tensorboard: bool
	_record_obs: Set[str]
	_agents_models: List[Callable]
	
	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int],
				 buffer_size: int, gamma: float, observation_space: Space, agent_models: List[Callable], agent_list: List[str],
				 use_gpu: bool, handle_timeout: bool, use_tensorboard: bool = False, tensorboard_data: List = None):
		"""
		Initialize the single controllable agent multi-agent scneario DQN
		
		:param num_agents: number of agents in the environment
		:param action_dim: number of actions of the agent, the DQN is agnostic to the semantic of each action
        :param num_layers: number of layers for the q_network
        :param act_function: activation function for the q_network
        :param layer_sizes: number of neurons in each layer (list must have equal number of entries as the number of layers)
        :param buffer_size: buffer size for the replay buffer
        :param gamma: reward discount factor
        :param observation_space: gym space for the agent observations
		:param agent_models: list with the prediction functions for the not-controllable agents
		:param agent_list: list with the ids of the agents in the environment, controlled agent must have the id "controlled"
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
        :type agent_list: list[str]
        :type use_gpu: bool
        :type handle_timeout: bool
        :type use_tensorboard: bool
        :type gamma: float
        :type act_function: callable
        :type agent_models: list[callable]
        :type observation_space: gym.Space
        :type tensorboard_data: list
		"""
		self._num_agents = num_agents
		self._write_tensorboard = use_tensorboard
		self._record_obs = []
		self._agent_dqn = DQNetwork(action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, observation_space, use_gpu, handle_timeout,
									use_tensorboard, tensorboard_data)
		self._agents_models = agent_models
		self._agent_list = agent_list.copy()
	
	########################
	### Class Properties ###
	########################
	@property
	def num_agents(self) -> int:
		return self._num_agents
	
	@property
	def agent_dqns(self) -> DQNetwork:
		return self._agent_dqn
	
	@property
	def write_tensorboard(self) -> bool:
		return self._write_tensorboard
	
	@num_agents.setter
	def num_agents(self, num_agents: int):
		self._num_agents = num_agents
	
	@agent_dqns.setter
	def agent_dqns(self, agent_dqn: DQNetwork):
		self._agent_dqn = agent_dqn
	
	@write_tensorboard.setter
	def write_tensorboard(self, write_tensorboard: bool):
		self._write_tensorboard = write_tensorboard
	
	#####################
	### Class Methods ###
	#####################
	def train_dqns(self, env: gymnasium.Env, total_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
				   eps_type: str, rng_seed: int, log_filename: str, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1,
				   target_freq: int = 100, tensorboard_frequency: int = 1):
		
		def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, epoch: int, max_timestpes: int):
			
			if update_type == 1:
				return max(((final_eps - init_eps) / max_timestpes) * COEF * epoch + init_eps, end_eps)
			elif update_type == 2:
				return max(decay_rate ** epoch * init_eps, end_eps)
			elif update_type == 3:
				return max((1 / (1 + decay_rate * epoch)) * init_eps, end_eps)
			elif update_type == 4:
				return max((decay_rate * math.sqrt(epoch)) * init_eps, end_eps)
			else:
				print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
				return max((1 / (1 + decay_rate * epoch)) * init_eps, end_eps)
		
		random.seed(rng_seed)
		np.random.seed(rng_seed)
		key = jax.random.PRNGKey(rng_seed)
		key, q_key = jax.random.split(key, 2)
		# train_log_file = Path(__file__).parent.absolute().parent.absolute().parent.absolute() / 'logs' / log_filename
		
		# Setup DQNs for training
		obs, _ = env.reset()
		
		if self._agent_dqn.online_state is None:
			self._agent_dqn.online_state = TrainState.create(
				apply_fn=self._agent_dqn.q_network.apply,
				params=self._agent_dqn.q_network.init(q_key, obs[0]),
				tx=optax.adam(learning_rate=optim_learn_rate),
			)
		
		if self._agent_dqn.target_params is None:
			self._agent_dqn.target_params = self._agent_dqn.q_network.init(q_key, obs[0])
		
		self._agent_dqn.q_network.apply = jax.jit(self._agent_dqn.q_network.apply)
		self._agent_dqn.target_params = optax.incremental_update(self._agent_dqn.online_state.params, self._agent_dqn.target_params, 1.0)
		
		start_time = time.time()
		to_convergence = (total_timesteps < 0)
		done = False
		epoch = 0
		loss = math.inf
		sys.stdout.flush()
		
		while not done:
			print("Epoch %d out of %d" % (epoch + 1, total_timesteps))
			
			# interact with environment
			self._record_obs.add(' '.join([str(x) for x in obs]))
			eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, total_timesteps)
			print('Epsilon: %f' % eps)
			if random.random() < eps:
				actions = np.array(env.action_space.sample())
			else:
				actions = []
				model_idx = 0
				for a_idx in range(self._num_agents):
					if self._agent_list[a_idx].find('controlled') != -1:
						q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs)
						action = q_values.argmax(axis=-1)
						action = jax.device_get(action)
						actions += [action]
					else:
						actions += [self._agents_models[model_idx](obs)]
						model_idx += 1
				actions = np.array(actions)
			next_obs, rewards, dones, infos, _ = env.step(actions)
			
			if len(rewards) == 1:
				rewards = [rewards] * self._num_agents
			
			if len(dones) == 1:
				dones = [dones] * self._num_agents
			
			# if self._write_tensorboard and all(dones):
			# 	self._agent_dqn.summary_writer.add_scalar("charts/episodic_return", infos["episode"]["r"], epoch)
			# 	self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", infos["episode"]["l"], epoch)
			# 	self._agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, epoch)
			
			# store new samples
			real_next_obs = list(next_obs).copy()
			for a_idx in range(self._num_agents):
				# for idx, d in enumerate(dones):
				# 	if d:
				# 		real_next_obs[a_idx][idx] = infos[idx]["terminal_observation"]
				self._agent_dqn.replay_buffer.add(obs[a_idx], real_next_obs[a_idx], actions[a_idx], rewards[a_idx], dones[a_idx], infos)
			obs = next_obs
			
			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					train_info = ('epoch: %d \t' % epoch)
					losses = []
					for a_idx in range(self._num_agents):
						device = self._agent_dqn.replay_buffer.device
						data = self._agent_dqn.replay_buffer.sample(batch_size)
						
						# perform a gradient-descent step
						if device == 'cpu':
							loss, old_val, self._agent_dqn.online_state = self._agent_dqn.compute_loss(
								self._agent_dqn.online_state,
								data.observations.numpy(),
								data.actions.numpy(),
								data.next_observations.numpy(),
								data.rewards.flatten().numpy(),
								data.dones.flatten().numpy()
							)
						else:
							loss, old_val, self._agent_dqn.online_state = self._agent_dqn.compute_loss(
								self._agent_dqn.online_state,
								data.observations.cpu().numpy(),
								data.actions.cpu().numpy(),
								data.next_observations.cpu().numpy(),
								data.rewards.flatten().cpu().numpy(),
								data.dones.flatten().cpu().numpy()
							)
						
						losses += [loss]
						train_info += ('agent loss: %.7f\t' % loss)
						
						# write log data for tensorboard
						if self._write_tensorboard and epoch % tensorboard_frequency == 0:
							self._agent_dqn.summary_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
							self._agent_dqn.summary_writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), epoch)
							print("SPS:", int(epoch / (time.time() - start_time)))
							self._agent_dqn.summary_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
					
					print('Train losses: ' + ','.join([str(x) for x in losses]))
					print('Train Info: ' + train_info)
				
				if epoch % target_freq == 0:
					self._agent_dqn.target_params = optax.incremental_update(self._agent_dqn.online_state.params, self._agent_dqn.target_params, tau)
			
			if (to_convergence and abs(loss) < 1e-9) or (not to_convergence and epoch >= total_timesteps):
				done = True
			
			epoch += 1
			sys.stdout.flush()
			if np.all(dones):
				obs, _ = env.reset()
	
	def save_model(self, filename: str, model_dir: Path) -> None:
		model_path = model_dir / (filename + '_centralized.model')
		obs_path = model_dir / (filename + '_centralized.json')
		with open(model_path, "wb") as mf:
			mf.write(flax.serialization.to_bytes(self._agent_dqn.online_state))
		with open(obs_path, "w") as f:
			f.write(json.dumps(self._record_obs))
		print("Model state saved to file: " + str(model_path))
	
	def load_model(self, filename: str, model_dir: Path) -> None:
		file_path = model_dir / filename
		template = TrainState.create(apply_fn=self._agent_dqn.q_network.apply,
									 params=self._agent_dqn.q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7))),
									 tx=optax.adam(learning_rate=0.0001))
		with open(file_path, "rb") as f:
			self._agent_dqn.online_state = flax.serialization.from_bytes(template, f.read())
		print("Loaded model state from file: " + str(file_path))
	
	def load_obs(self, filename: str, model_dir: Path) -> None:
		file_path = model_dir / filename
		with open(file_path, "r") as f:
			self._record_obs = set(json.load(f))
		print("Loaded model state from file: " + str(file_path))
	
	def get_policy(self) -> np.ndarray:
		
		pol = []
		for rec_obs in self._record_obs:
			obs = np.array([float(x) for x in rec_obs.split(' ')])
			q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs)
			max_q = q_values.max(axis=-1)
			tmp_pol = np.isclose(q_values, max_q, rtol=1e-10, atol=1e-10).astype(int)
			pol += [tmp_pol / sum(tmp_pol)]
		
		return pol
	
	
class LegibleCentralMADQN(CentralizedTrainingMADQN):

	_optimal_models: Dict[str, TrainState]
	_goal_ids: List[str]
	_goal: str

	def __init__(self, num_agents: int, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
				 observation_space: Space, use_gpu: bool, handle_timeout: bool, models_dir: Path, optimal_filenames: List[str], goal_ids: List[str], goal: str,
				 use_tensorboard: bool = False, tensorboard_data: List = None):
		
		super().__init__(num_agents, action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma, observation_space, use_gpu, handle_timeout,
						 use_tensorboard, tensorboard_data)
		
		self._goal_ids = goal_ids.copy()
		self._goal = goal
		for goal_id in goal_ids:
			idx = goal_ids.index(goal_id)
			file_path = models_dir / optimal_filenames[idx]
			template = TrainState.create(apply_fn=self._agent_dqn.q_network.apply,
										 params=self._agent_dqn.q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7)), train=False),
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
	def train_dqn(self, env: gymnasium.Env, total_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
				  eps_type: str, rng_seed: int, log_filename: str, exploration_decay: float = 0.99, warmup: int = 0, train_freq: int = 1,
				  target_freq: int = 100,
				  tensorboard_frequency: int = 1):
		
		def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, epoch: int, max_timestpes: int):
			
			if update_type == 1:
				return max(((final_eps - init_eps) / max_timestpes) * COEF * epoch + init_eps, end_eps)
			elif update_type == 2:
				return max(decay_rate ** epoch * init_eps, end_eps)
			elif update_type == 3:
				return max((1 / (1 + decay_rate * epoch)) * init_eps, end_eps)
			elif update_type == 4:
				return max((decay_rate * math.sqrt(epoch)) * init_eps, end_eps)
			else:
				print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
				return max((1 / (1 + decay_rate * epoch)) * init_eps, end_eps)
		
		random.seed(rng_seed)
		np.random.seed(rng_seed)
		key = jax.random.PRNGKey(rng_seed)
		key, q_key = jax.random.split(key, 2)
		# train_log_file = Path(__file__).parent.absolute().parent.absolute().parent.absolute() / 'logs' / log_filename
		
		# Setup DQNs for training
		obs, _ = env.reset()
		
		if self._agent_dqn.online_state is None:
			self._agent_dqn.online_state = TrainState.create(
				apply_fn=self._agent_dqn.q_network.apply,
				params=self._agent_dqn.q_network.init(q_key, obs[0]),
				tx=optax.adam(learning_rate=optim_learn_rate),
			)
		
		if self._agent_dqn.target_params is None:
			self._agent_dqn.target_params = self._agent_dqn.q_network.init(q_key, obs[0])
		
		self._agent_dqn.q_network.apply = jax.jit(self._agent_dqn.q_network.apply)
		self._agent_dqn.target_params = optax.incremental_update(self._agent_dqn.online_state.params, self._agent_dqn.target_params, 1.0)
		
		start_time = time.time()
		to_convergence = (total_timesteps < 0)
		done = False
		epoch = 0
		loss = math.inf
		sys.stdout.flush()
		
		while not done:
			print("Epoch %d out of %d" % (epoch + 1, total_timesteps))
			
			# interact with environment
			for a_idx in range(self._num_agents):
				self._record_obs.add(' '.join([str(x) for x in obs[a_idx]]))
			eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, total_timesteps)
			if random.random() < eps:
				actions = np.array(env.action_space.sample())
			else:
				actions = []
				for a_idx in range(self._num_agents):
					q_values = self._agent_dqn.q_network.apply(self._agent_dqn.online_state.params, obs[a_idx])
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
			next_obs, _, dones, infos, _ = env.step(actions)
			
			# Obtain the legible rewards
			legible_rewards = []
			for a_idx in range(self._num_agents):
				act_q_vals = []
				goal_q = 0.0
				action = actions[a_idx]
				for goal in self._goal_ids:
					act_q_vals += [self._agent_dqn.q_network.apply(self._optimal_models[goal].network_state.params, obs[a_idx])[action]]
					if goal == self._goal:
						goal_q = act_q_vals[-1]
				legible_rewards += [goal_q / sum(act_q_vals)]
			
			if len(legible_rewards) == 1:
				legible_rewards = [legible_rewards] * self._num_agents
			
			if len(dones) == 1:
				dones = [dones] * self._num_agents
			
			if self._write_tensorboard and "final_info" in infos:
				for info in infos["final_info"]:
					if "episode" in info.keys():
						print(f"global_step={epoch}, episodic_return={info['episode']['r']}")
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_return", info["episode"]["r"], epoch)
						self._agent_dqn.summary_writer.add_scalar("charts/episodic_length", info["episode"]["l"], epoch)
						self._agent_dqn.summary_writer.add_scalar("charts/epsilon", eps, epoch)
						break
			
			# store new samples
			real_next_obs = list(next_obs).copy()
			for a_idx in range(self._num_agents):
				self._agent_dqn.replay_buffer.add(obs[a_idx], real_next_obs[a_idx], actions[a_idx], legible_rewards[a_idx], dones[a_idx], infos)
			obs = next_obs
			
			# update Q-network and target network
			if epoch >= warmup:
				if epoch % train_freq == 0:
					train_info = ('epoch: %d \t' % epoch)
					losses = []
					for a_idx in range(self._num_agents):
						device = self._agent_dqn.replay_buffer.device
						data = self._agent_dqn.replay_buffer.sample(batch_size)
						
						# perform a gradient-descent step
						if device == 'cpu':
							loss, old_val, self._agent_dqn.online_state = self._agent_dqn.compute_loss(
								self._agent_dqn.online_state,
								data.observations.numpy(),
								data.actions.numpy(),
								data.next_observations.numpy(),
								data.rewards.flatten().numpy(),
								data.dones.flatten().numpy()
							)
						else:
							loss, old_val, self._agent_dqn.online_state = self._agent_dqn.compute_loss(
								self._agent_dqn.online_state,
								data.observations.cpu().numpy(),
								data.actions.cpu().numpy(),
								data.next_observations.cpu().numpy(),
								data.rewards.flatten().cpu().numpy(),
								data.dones.flatten().cpu().numpy()
							)
						
						losses += [loss]
						train_info += ('agent loss: %.7f\t' % loss)
						
						# write log data for tensorboard
						if self._write_tensorboard and epoch % tensorboard_frequency == 0:
							self._agent_dqn.summary_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
							self._agent_dqn.summary_writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), epoch)
							print("SPS:", int(epoch / (time.time() - start_time)))
							self._agent_dqn.summary_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
					
					# print('Train losses: ' + ','.join([str(x) for x in losses]))
					print('Train Info: ' + train_info)
				
				if epoch % target_freq == 0:
					self._agent_dqn.target_params = optax.incremental_update(self._agent_dqn.online_state.params, self._agent_dqn.target_params, tau)
			
			if (to_convergence and abs(loss) < 1e-9) or (not to_convergence and epoch >= total_timesteps):
				done = True
			
			epoch += 1
			sys.stdout.flush()
			if np.all(dones):
				obs, _ = env.reset()
				
		