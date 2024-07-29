#! /usr/bin/env python

import jax
import jax.numpy as jnp
import flax
import optax
import flax.linen as nn
import numpy as np
import logging
import wandb

from typing import Callable, Tuple, Union, List, Optional
from flax.training.train_state import TrainState
from flax.training.checkpoints import save_checkpoint
from algos.q_networks import CNNCriticNetwork, CriticNetwork, CNNActorNetwork, ActorNetwork
from utilities.buffers import ReplayBuffer
from pathlib import Path
from functools import partial


class DDPG(object):

	_critic_network: nn.Module
	_actor_network: nn.Module
	_critic_online_state: Optional[TrainState]
	_actor_online_state: Optional[TrainState]
	_critic_target_params: flax.core.FrozenDict
	_actor_target_params: flax.core.FrozenDict
	_wandb_writer: wandb.run
	_use_cnn: bool
	
	def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], action_bias: jnp.ndarray, action_scale: jnp.ndarray,
				 cnn_layer: bool = False, use_wandb: bool = False, cnn_properties: List = None, wandb_writer: wandb.run = None, action_bounds: Tuple[float, float] = (-1.0, 1.0)):
		
		if cnn_layer:
			self._use_cnn = True
			if cnn_properties is None:
				n_conv_layers = 1
				cnn_size = 128
				cnn_kernel = (3, 3)
				pool_window = (2, 2)
			else:
				n_conv_layers = cnn_properties[0]
				cnn_size = cnn_properties[1]
				cnn_kernel = cnn_properties[2]
				pool_window = cnn_properties[3]
			self._critic_network = CNNCriticNetwork(action_dim=action_dim, num_linear_layers=num_layers, activation_function=act_function,
													layer_sizes=layer_sizes.copy(), num_conv_layers=n_conv_layers, cnn_size=cnn_size, cnn_kernel=cnn_kernel,
													pool_window=pool_window)
			self._actor_network = CNNActorNetwork(action_dim=action_dim, num_linear_layers=num_layers, activation_function=act_function,
												  layer_sizes=layer_sizes.copy(), num_conv_layers=n_conv_layers, cnn_size=cnn_size, cnn_kernel=cnn_kernel,
												  pool_window=pool_window, action_bias=action_bias, action_scale=action_scale, action_bounds=action_bounds)
		
		else:
			self._use_cnn = False
			self._critic_network = CriticNetwork(action_dim=action_dim, num_layers=num_layers, activation_function=act_function, layer_sizes=layer_sizes.copy())
			self._actor_network = ActorNetwork(action_dim=action_dim, num_layers=num_layers, activation_function=act_function, layer_sizes=layer_sizes.copy(),
			                                   action_bias=action_bias, action_scale=action_scale, action_bounds=action_bounds)
		
		self._actor_network.apply = jax.jit(self._actor_network.apply)
		self._critic_network.apply = jax.jit(self._critic_network.apply)
		self._initialized = False
		self._actor_online_state = None
		self._critic_online_state = None
		self._actor_target_params = flax.core.FrozenDict({})
		self._critic_target_params = flax.core.FrozenDict({})
		
		self._use_wandb = use_wandb
		if use_wandb:
			self._wandb_writer = wandb_writer
	
	#########################
	### GETTERS & SETTERS ###
	#########################
	@property
	def actor_network(self) -> nn.Module:
		return self._actor_network
	
	@property
	def critic_network(self) -> nn.Module:
		return self._critic_network
	
	@property
	def critic_online_state(self) -> TrainState:
		return self._critic_online_state
	
	@property
	def actor_online_state(self) -> TrainState:
		return self._actor_online_state
	
	@property
	def wandb_writer(self) -> wandb.run:
		return self._wandb_writer
	
	@property
	def use_tensorboard(self) -> bool:
		return self._use_wandb
	
	###################
	### Class Utils ###
	###################
	def init_networks(self, rng_seed: int, obs: Union[np.ndarray, jnp.ndarray], acts: Union[np.ndarray, jnp.ndarray], actor_lr: float, critic_lr: float):
		key = jax.random.PRNGKey(rng_seed)
		key, q_key = jax.random.split(key, 2)
		if not self._initialized:
			# Initialize online params
			self._actor_online_state = TrainState.create(
				params=self._actor_network.init(q_key, obs),
				apply_fn=self._actor_network.apply,
				tx=optax.adam(learning_rate=actor_lr))
			self._critic_online_state = TrainState.create(
				params=self._critic_network.init(q_key, obs, acts),
				apply_fn=self._critic_network.apply,
				tx=optax.adam(learning_rate=critic_lr))
			
			# Initialize target params
			actor_target_params = self._actor_network.init(q_key, obs)
			update_target_state_params = optax.incremental_update(self._actor_online_state.params, actor_target_params, 1.0)
			self._actor_target_params = flax.core.freeze(update_target_state_params)
			critic_target_params = self._critic_network.init(q_key, obs, acts)
			update_target_state_params = optax.incremental_update(self._critic_online_state.params, critic_target_params, 1.0)
			self._critic_target_params = flax.core.freeze(update_target_state_params)
		
		self._initialized = True
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_critic_loss(self, q_state: TrainState, observations: Union[np.ndarray, jax.Array], actions: Union[np.ndarray, jax.Array],
							next_observations: Union[np.ndarray, jax.Array], rewards: Union[np.ndarray, jax.Array],
							dones: Union[np.ndarray, jax.Array], gamma: float) -> Tuple[float, TrainState]:
		
		def mse_loss(params: flax.core.FrozenDict):
			q = self._critic_network.apply(params, observations, actions).squeeze()             								# get online model's q_values
			return ((q - next_q_value) ** 2).mean()

		next_actions = self._actor_network.apply(self._actor_target_params, next_observations)               	                # get actor's actions
		q_next = self._critic_network.apply(self._critic_target_params, next_observations, next_actions)			            # get critic evaluation
		next_q_value = (rewards + (1 - dones) * gamma * q_next).reshape(-1)															# compute Bellman equation

		loss_value, grads = jax.value_and_grad(mse_loss)(q_state.params)
		q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_state
	
	@partial(jax.jit, static_argnums=(0,))
	def compute_actor_loss(self, q_state: TrainState, observations: Union[np.ndarray, jax.Array]) -> Tuple[float, TrainState]:
		
		def loss(params: flax.core.FrozenDict):
			return -self._critic_network.apply(self._critic_online_state.params, observations, self._actor_network.apply(params, observations)).mean()
		
		loss_value, grads = jax.value_and_grad(loss)(q_state.params)
		q_state = q_state.apply_gradients(grads=grads)
		return loss_value, q_state
	
	def update_models(self, replay_buffer: ReplayBuffer, batch_size: int, gamma: float, cnn_shape: Tuple[int] = None) -> Tuple[float, float]:
		
		data = replay_buffer.sample(batch_size)
		observations = data.observations
		next_observations = data.next_observations
		actions = data.actions
		rewards = data.rewards
		dones = data.dones
		
		if self._use_cnn and cnn_shape is not None:
			observations = observations.reshape((*observations.shape[:2], *cnn_shape))
			next_observations = next_observations.reshape((*next_observations.shape[:2], *cnn_shape))
			
		critic_loss, self._critic_online_state = self.compute_critic_loss(self._critic_online_state, observations, actions, next_observations, rewards,
																			 dones, gamma)
		actor_loss, self._actor_online_state = self.compute_actor_loss(self._actor_online_state, observations)

		return critic_loss, actor_loss
		
	def update_targets(self, tau: float) -> None:
		new_actor_targets = optax.incremental_update(self._actor_online_state.params, self._actor_target_params.unfreeze(), tau)
		self._actor_target_params = flax.core.freeze(new_actor_targets)
		
		new_critic_targets = optax.incremental_update(self._critic_online_state.params, self._critic_target_params.unfreeze(), tau)
		self._critic_target_params = flax.core.freeze(new_critic_targets)
  
	def create_checkpoint(self, model_dir: Path, epoch: int = 0) -> None:
		save_checkpoint(ckpt_dir=model_dir, target=self._actor_online_state, step=epoch)
		save_checkpoint(ckpt_dir=model_dir, target=self._critic_online_state, step=epoch)
	
	def save_model(self, filename: str, model_dir: Path, logger: logging.Logger) -> None:
		file_path = model_dir / (filename + '_actor.model')
		with open(file_path, "wb") as f:
			f.write(flax.serialization.to_bytes(self._actor_online_state))
		logger.info("Actor model state saved to file: " + str(file_path))
		
		file_path = model_dir / (filename + '_critic.model')
		with open(file_path, "wb") as f:
			f.write(flax.serialization.to_bytes(self._critic_online_state))
		logger.info("Critic model state saved to file: " + str(file_path))