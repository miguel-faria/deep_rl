#! /usr/bin/env python

import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, List, Tuple

class QNetwork(nn.Module):
	action_dim: int
	num_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray) -> jnp.ndarray:
		x = jnp.array(x_orig)
		for i in range(self.num_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i], dtype=jnp.float32)(x))
		return nn.Dense(self.action_dim)(x)


class DuelingQNetwork(nn.Module):
	action_dim: int
	num_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray) -> jnp.ndarray:
		x = jnp.array(x_orig)
		for i in range(self.num_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i], dtype=jnp.float32)(x))
		a = nn.Dense(self.action_dim)(x)
		v = nn.Dense(1)(x)
		return v + (a - a.mean())
	

class CNNQNetwork(nn.Module):
	action_dim: int
	num_linear_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	num_conv_layers: int
	cnn_size: List[int]
	cnn_kernel: List[Tuple[int]]
	pool_window: List[Tuple[int]]
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray) -> jnp.ndarray:
		x = x_orig
		for i in range(self.num_conv_layers):
			x = self.activation_function(nn.Conv(self.cnn_size[i], kernel_size=self.cnn_kernel[i], padding='SAME', dtype=jnp.float32)(x))
			x = nn.max_pool(x, window_shape=self.pool_window[i], padding='VALID')
		x = x.reshape((x.shape[0], -1))
		for i in range(self.num_linear_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i], dtype=jnp.float32)(x))
		return nn.Dense(self.action_dim)(x)


class CNNDuelingQNetwork(nn.Module):
	action_dim: int
	num_linear_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	num_conv_layers: int
	cnn_size: List[int]
	cnn_kernel: List[Tuple[int]]
	pool_window: List[Tuple[int]]
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray) -> jnp.ndarray:
		x = x_orig
		for i in range(self.num_conv_layers):
			x = self.activation_function(nn.Conv(self.cnn_size[i], kernel_size=self.cnn_kernel[i], padding='SAME', dtype=jnp.float32)(x))
			x = nn.max_pool(x, window_shape=self.pool_window[i], padding='VALID')
		x = x.reshape((x.shape[0], -1))
		for i in range(self.num_linear_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i], dtype=jnp.float32)(x))
		a = nn.Dense(self.action_dim)(x)
		v = nn.Dense(1)(x)
		return v + (a - a.mean())
		# return v + (a - a.max())


class MultiObsCNNDuelingQNetwork(nn.Module):
	action_dim: int
	num_obs: int
	num_linear_layers: int
	layer_sizes: List[int]
	activation_function: Callable
	num_conv_layers: int
	cnn_size: List[int]
	cnn_kernel: List[Tuple[int]]
	pool_window: List[Tuple[int]]
	
	@nn.compact
	def __call__(self, x_orig: jnp.ndarray) -> jnp.ndarray:
		x = jnp.array([])
		for idx in range(self.num_obs):
			x_temp = x_orig[:, idx]
			for i in range(self.num_conv_layers):
				x_temp = self.activation_function(nn.Conv(self.cnn_size[i], kernel_size=self.cnn_kernel[i], padding='SAME', dtype=jnp.float32)(x_temp))
				x_temp = nn.max_pool(x_temp, window_shape=self.pool_window[i], padding='VALID')
			x = jnp.append(x, x_temp).reshape((x_orig.shape[0], -1))
		for i in range(self.num_linear_layers):
			x = self.activation_function(nn.Dense(self.layer_sizes[i], dtype=jnp.float32)(x))
		a = nn.Dense(self.action_dim)(x)
		v = nn.Dense(1)(x)
		return v + (a - a.mean())