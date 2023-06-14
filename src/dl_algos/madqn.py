#! /usr/bin/env python
import math
import random
import time
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete, Space
from typing import Callable, List
from pathlib import Path
from termcolor import colored
from dl_algos.dqn import DQNetwork, EPS_TYPE
from typing import List, Dict, Callable, Set


class MultiAgentDQN(object):
	
	_num_agents: int
	_agent_ids: List[str]
	_agent_dqns: Dict[str, DQNetwork]
	_write_tensorboard: bool
	_record_obs: List[Set[str]]
	
	def __init__(self, num_agents: int, agent_ids: List[str], action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int],
				 buffer_size: int, gamma: float, observation_space: Space, use_gpu: bool, handle_timeout: bool, use_tensorboard: bool = False,
				 tensorboard_data: List = None):
		
		self._num_agents = num_agents
		self._agent_ids = agent_ids
		self._write_tensorboard = use_tensorboard
		
		self._agent_dqns = {}
		self._record_obs = []
		for agent_id in agent_ids:
			self._agent_dqns[agent_id] = DQNetwork(action_dim, num_layers, act_function, layer_sizes, buffer_size, gamma,
												   observation_space, use_gpu, handle_timeout, use_tensorboard, tensorboard_data)
			self._record_obs += [set()]
	
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
	def train_dqns(self, env: gym.Env, total_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float, final_eps: float,
				   eps_type: str, rng_seed: int, log_filename: str, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 100, train_freq: int = 1,
				   tensorboard_frequency: int = 1):
		
		def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, epoch: int, max_timestpes: int):
			
			if update_type == 1:
				return max(((final_eps - init_eps) / max_timestpes) * epoch + init_eps, end_eps)
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
		train_log_file = Path(__file__).parent.absolute().parent.absolute().parent.absolute() / 'logs' / log_filename
		
		# Setup DQNs for training
		obs, _, _, _ = env.reset()
		
		for a_id in self._agent_ids:
			agent_dqn = self._agent_dqns[a_id]
			a_idx = self._agent_ids.index(a_id)
			if agent_dqn.network_state is None:
				agent_dqn.network_state = TrainState.create(
					apply_fn=agent_dqn.q_network.apply,
					params=agent_dqn.q_network.init(q_key, obs[a_idx]),
					tx=optax.adam(learning_rate=optim_learn_rate),
				)
			
			if agent_dqn.target_params is None:
				agent_dqn.target_params = agent_dqn.q_network.init(q_key, obs[a_idx])
			
			agent_dqn.q_network.apply = jax.jit(agent_dqn.q_network.apply)
			agent_dqn.target_params = optax.incremental_update(agent_dqn.network_state.params, agent_dqn.target_params, 1.0)
		
		start_time = time.time()
		to_convergence = (total_timesteps < 0)
		done = False
		epoch = 0
		loss = math.inf
		
		while not done:
			print("Epoch %d out of %d" % (epoch + 1, total_timesteps))
			
			# interact with environment
			for a_id in self._agent_ids:
				a_idx = self._agent_ids.index(a_id)
				self._record_obs[a_idx].add(' '.join([str(x) for x in obs[a_idx]]))
			eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, total_timesteps)
			if random.random() < eps:
				actions = np.array(env.action_space.sample())
			else:
				actions = []
				for a_id in self._agent_ids:
					a_idx = self._agent_ids.index(a_id)
					q_values = self._agent_dqns[a_id].q_network.apply(self._agent_dqns[a_id].network_state.params, obs[a_idx])
					action = q_values.argmax(axis=-1)
					action = jax.device_get(action)
					actions += [action]
				actions = np.array(actions)
			next_obs, rewards, dones, infos = env.step(actions)
			
			if len(rewards) == 1:
				rewards = [rewards] * self._num_agents
			
			if len(dones) == 1:
				dones = [dones] * self._num_agents
			
			if self._write_tensorboard:
				for info in infos:
					if "episode" in info.keys():
						print(f"global_step={epoch}, episodic_return={info['episode']['r']}")
						for a_id in self._agent_ids:
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/episodic_return", info["episode"]["r"], epoch)
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/episodic_length", info["episode"]["l"], epoch)
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/epsilon", eps, epoch)
						break
			
			# store new samples
			real_next_obs = list(next_obs).copy()
			for a_id in self._agent_ids:
				a_idx = self._agent_ids.index(a_id)
				for idx, d in enumerate(dones):
					if d:
						real_next_obs[a_idx][idx] = infos[idx]["terminal_observation"]
				self._agent_dqns[a_id].replay_buffer.add(obs[a_idx], real_next_obs[a_idx], actions[a_idx], rewards[a_idx], dones[a_idx], infos)
				obs = next_obs
			
			# update Q-network and target network
			if epoch > warmup:
				if epoch % train_freq == 0:
					train_info = ('epoch: %d \t' % epoch)
					losses = []
					for a_id in self._agent_ids:
						data = self._agent_dqns[a_id].replay_buffer.sample(batch_size)
						
						# perform a gradient-descent step
						loss, old_val, self._agent_dqns[a_id].network_state = self._agent_dqns[a_id].update(
							self._agent_dqns[a_id].network_state,
							data.observations.numpy(),
							data.actions.numpy(),
							data.next_observations.numpy(),
							data.rewards.flatten().numpy(),
							data.dones.flatten().numpy(),
						)
						
						losses += [loss]
						train_info += ('agent %s: loss: %.7f\t' % (a_id, loss))
						
						# write log data for tensorboard
						if self._write_tensorboard and epoch % tensorboard_frequency == 0:
							self._agent_dqns[a_id].summary_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
							self._agent_dqns[a_id].summary_writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), epoch)
							print("SPS:", int(epoch / (time.time() - start_time)))
							self._agent_dqns[a_id].summary_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
					
					print('Train losses: ' + ','.join([str(x) for x in losses]))
					print('Train Info: ' + train_info)
				
				if epoch % target_freq == 0:
					for a_id in self._agent_ids:
						self._agent_dqns[a_id].target_params = optax.incremental_update(self._agent_dqns[a_id].network_state.params,
																						self._agent_dqns[a_id].target_params, tau)
			
				if to_convergence and abs(loss) < 1e-9 or not to_convergence and epoch >= total_timesteps:
					done = True
			
			epoch += 1
		
	def save_models(self, filename: str, model_dir: Path) -> None:
		for agent_id in self._agent_ids:
			file_path = model_dir / (filename + '_' + agent_id + '.model')
			with open(file_path, "wb") as f:
				f.write(flax.serialization.to_bytes(self._agent_dqns[agent_id].network_state))
		print("Model states saved to files")
	
	def save_model(self, filename: str, agent_id: str, model_dir: Path) -> None:
		file_path = model_dir / (filename + '_' + agent_id + '.model')
		with open(file_path, "wb") as f:
			f.write(flax.serialization.to_bytes(self._agent_dqns[agent_id].network_state))
		print("Model state saved to file: " + str(file_path))
	
	def load_models(self, filename_prefix: str, model_dir: Path) -> None:
		for agent_id in self._agent_ids:
			file_path = model_dir / (filename_prefix + '_' + agent_id + '.model')
			template = TrainState.create(apply_fn=self._agent_dqns[agent_id].q_network.apply,
										 params=self._agent_dqns[agent_id].q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7)), train=False),
										 tx=optax.adam(learning_rate=0.0001))
			with open(file_path, "rb") as f:
				self._agent_dqns[agent_id].network_state = flax.serialization.from_bytes(template, f.read())
			print("Loaded model states from files")
	
	def load_model(self, filename: str, agent_id: str, model_dir: Path) -> None:
		file_path = model_dir / filename
		template = TrainState.create(apply_fn=self._agent_dqns[agent_id].q_network.apply,
									 params=self._agent_dqns[agent_id].q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7)), train=False),
									 tx=optax.adam(learning_rate=0.0001))
		with open(file_path, "rb") as f:
			self._agent_dqns[agent_id].network_state = flax.serialization.from_bytes(template, f.read())
		print("Loaded model state from file: " + str(file_path))
	
	def get_policies(self) -> List[np.ndarray]:
		
		policies = []
		for a_id in self._agent_ids:
			a_pol = []
			a_idx = self._agent_ids.index(a_id)
			for rec_obs in self._record_obs[a_idx]:
				obs = np.array([float(x) for x in rec_obs.split(' ')])
				q_values = self._agent_dqns[a_id].q_network.apply(self._agent_dqns[a_id].network_state.params, obs)
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
			q_values = self._agent_dqns[a_id].q_network.apply(self._agent_dqns[a_id].network_state.params, obs)
			max_q = q_values.max(axis=-1)
			tmp_pol = np.isclose(q_values, max_q, rtol=1e-10, atol=1e-10).astype(int)
			pol += [tmp_pol / sum(tmp_pol)]
		
		return pol

		