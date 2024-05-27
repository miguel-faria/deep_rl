#! /usr/bin/env python

import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, List, Tuple


class CriticNetwork(nn.Module):
	action_dim: int
	num_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
		x = jnp.concatenate([x_orig, act], -1)
		for i in range(self.num_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i])(x))
		x = nn.Dense(1)(x)
		return x


class CNNCriticNetwork(nn.Module):
	action_dim: int
	num_linear_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	num_conv_layers: int
	cnn_size: List[int]
	cnn_kernel: List[Tuple[int]]
	pool_window: List[Tuple[int]]
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray, act: jnp.ndarray) -> jnp.ndarray:
		x = x_orig
		for i in range(self.num_conv_layers):
			x = self.activation_function(nn.Conv(self.cnn_size[i], kernel_size=self.cnn_kernel[i], padding='SAME')(x))
			x = nn.max_pool(x, window_shape=self.pool_window[i], padding='VALID')
		x = jnp.concatenate([x.reshape((x.shape[0], -1)), act], -1)
		for i in range(self.num_linear_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i])(x))
		x = nn.Dense(1)(x)
		return x


class ActorNetwork(nn.Module):
	action_dim: int
	num_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	action_scale: jnp.ndarray
	action_bias: jnp.ndarray
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray) -> jnp.ndarray:
		x = jnp.array(x_orig)
		for i in range(self.num_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i])(x))
		x = nn.tanh(nn.Dense(self.action_dim)(x))
		x = x * self.action_scale + self.action_bias
		return x


class CNNActorNetwork(nn.Module):
	action_dim: int
	num_linear_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	num_conv_layers: int
	cnn_size: List[int]
	cnn_kernel: List[Tuple[int]]
	pool_window: List[Tuple[int]]
	action_scale: jnp.ndarray
	action_bias: jnp.ndarray
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray) -> jnp.ndarray:
		x = x_orig
		for i in range(self.num_conv_layers):
			x = self.activation_function(nn.Conv(self.cnn_size[i], kernel_size=self.cnn_kernel[i], padding='SAME')(x))
			x = nn.max_pool(x, window_shape=self.pool_window[i], padding='VALID')
		x = x.reshape((x.shape[0], -1))
		for i in range(self.num_linear_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i])(x))
		x = nn.tanh(nn.Dense(self.action_dim)(x))
		x = x * self.action_scale + self.action_bias
		return x
