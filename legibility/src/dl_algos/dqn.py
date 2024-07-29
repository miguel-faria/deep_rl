#! /usr/bin/env python
import math
import pathlib
import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import logging

from dl_algos.q_networks import QNetwork, DuelingQNetwork, CNNQNetwork, CNNDuelingQNetwork, MultiObsCNNDuelingQNetwork
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.training.train_state import TrainState
from typing import Callable, List, Union, Optional
from pathlib import Path
from termcolor import colored
from functools import partial
from jax import jit
from wandb import run

EPS_TYPE = {'linear': 1, 'exp': 2, 'log': 3, 'epoch': 4}


class DQNetwork(object):

    _q_network: nn.Module
    _online_state: TrainState
    _target_state_params: flax.core.FrozenDict
    _gamma: float
    _use_ddqn: bool
    _cnn_layer: bool
    
    def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], gamma: float, dueling_dqn: bool = False,
                 use_ddqn: bool = False, cnn_layer: bool = False, cnn_properties: List = None, ma_obs: bool = False, n_obs: int = 1):
        
        """
        Initializes a DQN
        
        :param action_dim:          number of actions of the agent, the DQN is agnostic to the semantic of each action
        :param num_layers:          number of layers for the q_network
        :param act_function:        activation function for the q_network
        :param layer_sizes:         number of neurons in each layer (list must have equal number of entries as the number of layers)
        :param gamma:               reward discount factor
        :param dueling_dqn:         flag denoting the use of a dueling architecture
        :param use_ddqn:            flag denoting the use of a double dqn variant
        :param cnn_layer:           flag denoting the use of a convolutional layer as the entry layer
        :param cnn_properties:      list of the properties for the convolutional layer (layer size, kernel size, pooling window size)
        
        """
        
        if ma_obs:
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
            self._q_network = MultiObsCNNDuelingQNetwork(action_dim=action_dim, num_obs=n_obs, num_linear_layers=num_layers, activation_function=act_function,
                                                         layer_sizes=layer_sizes.copy(), num_conv_layers=n_conv_layers, cnn_size=cnn_size, cnn_kernel=cnn_kernel,
                                                         pool_window=pool_window)
        else:
            if cnn_layer:
                if cnn_properties is None:
                    n_conv_layers = 1
                    cnn_size = [128]
                    cnn_kernel = [(3, 3)]
                    pool_window = [(2, 2)]
                else:
                    n_conv_layers = cnn_properties[0]
                    cnn_size = cnn_properties[1]
                    cnn_kernel = cnn_properties[2]
                    pool_window = cnn_properties[3]
                if dueling_dqn:
                    self._q_network = CNNDuelingQNetwork(action_dim=action_dim, num_linear_layers=num_layers, activation_function=act_function,
                                                         layer_sizes=layer_sizes.copy(), num_conv_layers=n_conv_layers, cnn_size=cnn_size, cnn_kernel=cnn_kernel,
                                                         pool_window=pool_window)
                else:
                    self._q_network = CNNQNetwork(action_dim=action_dim, num_linear_layers=num_layers, activation_function=act_function,
                                                  layer_sizes=layer_sizes.copy(), num_conv_layers=n_conv_layers, cnn_size=cnn_size, cnn_kernel=cnn_kernel,
                                                  pool_window=pool_window)
            else:
                if dueling_dqn:
                    self._q_network = DuelingQNetwork(action_dim=action_dim, num_layers=num_layers, activation_function=act_function,
                                                      layer_sizes=layer_sizes.copy())
                else:
                    self._q_network = QNetwork(action_dim=action_dim, num_layers=num_layers, activation_function=act_function, layer_sizes=layer_sizes.copy())
            
        self._gamma = gamma
        self._target_state_params = None
        self._online_state = None
        self._use_ddqn = use_ddqn
        self._cnn_layer = cnn_layer
        self._q_network.apply = jax.jit(self._q_network.apply)
        self._dqn_initialized = False

    #############################
    ##    GETTERS & SETTERS    ##
    #############################

    @property
    def q_network(self) -> nn.Module:
        return self._q_network
    
    @property
    def online_state(self) -> TrainState:
        return self._online_state
    
    @property
    def target_params(self) -> flax.core.FrozenDict:
        return self._target_state_params
    
    @property
    def gamma(self) -> float:
        return self._gamma
    
    @property
    def cnn_layer(self) -> bool:
        return self._cnn_layer
    
    @property
    def dqn_initialized(self) -> bool:
        return self._dqn_initialized
    
    @dqn_initialized.setter
    def dqn_initialized(self, new_val: bool) -> None:
        self._dqn_initialized = new_val
    
    @gamma.setter
    def gamma(self, new_gamma: float) -> None:
        self._gamma = new_gamma
        
    @target_params.setter
    def target_params(self, new_params: flax.core.FrozenDict) -> None:
        self._target_state_params = new_params
        
    @online_state.setter
    def online_state(self, new_state: TrainState) -> None:
        self._online_state = new_state

    #############################
    ##       CLASS UTILS       ##
    #############################
    def init_network_states(self, rng_seed: int, obs: np.ndarray, optim_learn_rate: float, file_path: Union[str, Path] = ''):
        key = jax.random.PRNGKey(rng_seed)
        key, q_key = jax.random.split(key, 2)
        if self._online_state is None:
            if file_path == '':
                self._online_state = TrainState.create(
                    apply_fn=self._q_network.apply,
                    params=self._q_network.init(q_key, obs),
                    tx=optax.adam(learning_rate=optim_learn_rate),
                )
            else:
                template = TrainState.create(apply_fn=self._q_network.apply,
                                             params=self._q_network.init(q_key, obs),
                                             tx=optax.adam(learning_rate=optim_learn_rate))
                with open(file_path, "rb") as f:
                    self._online_state = flax.serialization.from_bytes(template, f.read())
        if self._target_state_params is None:
            self._target_state_params = self._q_network.init(q_key, obs)
            update_target_state_params = optax.incremental_update(self._online_state.params, self._target_state_params, 1.0)
            self._target_state_params = flax.core.freeze(update_target_state_params)
        
        self._dqn_initialized = True
    
    def compute_dqn_targets(self, dones: jnp.ndarray, next_observations: jnp.ndarray, rewards: jnp.ndarray,
                            target_state_params: flax.core.FrozenDict) -> Union[np.ndarray, jax.Array]:
        q_next_target = self._q_network.apply(target_state_params, next_observations)  # get target network q values
        q_next_target = jnp.max(q_next_target, axis=-1)  # get best q_val for each obs
        return rewards + (1 - dones) * self._gamma * q_next_target  # compute Bellman equation
    
    def compute_ddqn_targets(self, dones: jnp.ndarray, next_observations: jnp.ndarray, rewards: jnp.ndarray,
                             target_state_params: flax.core.FrozenDict, online_state_params: flax.core.FrozenDict) -> Union[np.ndarray, jax.Array]:
        q_next_target = self._q_network.apply(target_state_params, next_observations)  # get target network q values
        # DDQN the target network is used to evaluate how good the online network's predictions are
        q_next_online = self._q_network.apply(online_state_params, next_observations)  # get online network's prescribed actions
        online_acts = jnp.argmax(q_next_online, axis=1)
        q_next_target = q_next_target[np.arange(q_next_target.shape[0]), online_acts.squeeze()].reshape(-1, 1)  # get target's q values for prescribed actions
        return rewards + (1 - dones) * self._gamma * q_next_target  # compute Bellman equation
    
    def mse_loss(self, params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array], actions: Union[np.ndarray, jax.Array],
                 next_q_value: Union[np.ndarray, jax.Array]):
        q = self._q_network.apply(params, observations)  # get online model's q_values
        q = q[np.arange(q.shape[0]), actions.squeeze()].reshape(-1, 1)
        return ((q - next_q_value) ** 2).mean(), q  # compute loss
    
    @partial(jit, static_argnums=(0,))
    def compute_dqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array],
                         actions: Union[np.ndarray, jax.Array], next_observations: Union[np.ndarray, jax.Array], rewards: Union[np.ndarray, jax.Array],
                         dones: Union[np.ndarray, jax.Array]):
        next_q_value = self.compute_dqn_targets(dones, next_observations, rewards, target_state_params)
        
        (loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state
    
    @partial(jit, static_argnums=(0,))
    def compute_ddqn_loss(self, q_state: TrainState, target_state_params: flax.core.FrozenDict, observations: Union[np.ndarray, jax.Array],
                          actions: Union[np.ndarray, jax.Array], next_observations: Union[np.ndarray, jax.Array], rewards: Union[np.ndarray, jax.Array],
                          dones: Union[np.ndarray, jax.Array]):
        next_q_value = self.compute_ddqn_targets(dones, next_observations, rewards, target_state_params, q_state.params)
        
        (loss_value, q_pred), grads = jax.value_and_grad(self.mse_loss, has_aux=True)(q_state.params, observations, actions, next_q_value)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state
    
    @staticmethod
    def eps_update(update_type: int, init_eps: float, end_eps: float, decay_rate: float, step: int, max_steps: int):
        if update_type == 1:
            return max(((end_eps - init_eps) / max_steps) * step / decay_rate + init_eps, end_eps)
        elif update_type == 2:
            return max(decay_rate ** step * init_eps, end_eps)
        elif update_type == 3:
            return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)
        elif update_type == 4:
            return max((decay_rate * math.sqrt(step)) * init_eps, end_eps)
        else:
            print(colored('Unrecognized exploration decay type, defaulting to logarithmic decay', 'red'))
            return max((1 / (1 + decay_rate * step)) * init_eps, end_eps)

    def update_online_model(self, observations: jnp.ndarray, actions: jnp.ndarray, next_observations: jnp.ndarray, rewards: jnp.ndarray,
                            finished: jnp.ndarray) -> float:
        
        # perform a gradient-descent step
        if self._use_ddqn:
            td_loss, q_val, self._online_state = self.compute_ddqn_loss(self._online_state, self._target_state_params, observations, actions,
                                                                        next_observations, rewards, finished)
        else:
            td_loss, q_val, self._online_state = self.compute_dqn_loss(self._online_state, self._target_state_params, observations, actions,
                                                                       next_observations, rewards, finished)
        
        return float(td_loss)
    
    def update_target_model(self, tau: float):
        update_target_state_params = optax.incremental_update(self._online_state.params, self._target_state_params.unfreeze(), tau)
        self._target_state_params = flax.core.freeze(update_target_state_params)
    
    def get_action(self, obs):
        q_values = self._q_network.apply(self._q_network.variables, obs)
        actions = q_values.argmax()
        return jax.device_get(actions)
    
    def create_checkpoint(self, model_dir: Path, epoch: int = 0) -> None:
        save_checkpoint(ckpt_dir=model_dir, target=self._online_state, step=epoch)
    
    def load_checkpoint(self, ckpt_file: Path, logger: logging.Logger, obs_shape: tuple, epoch: int = -1) -> None:
        template = TrainState.create(apply_fn=self._q_network.apply,
                                     params=self._q_network.init(jax.random.PRNGKey(201), jnp.empty(obs_shape, dtype=jnp.float32)),
                                     tx=optax.adam(learning_rate=0.0))
        if epoch < 0:
            if pathlib.Path.is_file(ckpt_file):
                self._online_state = restore_checkpoint(ckpt_dir=ckpt_file, target=template)
            else:
                logger.error('ERROR!! Could not load checkpoint, expected checkpoint file got directory instead')
        else:
            if pathlib.Path.is_dir(ckpt_file):
                self._online_state = restore_checkpoint(ckpt_dir=ckpt_file, target=template, step=epoch)
            else:
                logger.error('ERROR!! Could not load checkpoint, expected checkpoint directory got file instead')
    
    def save_model(self, filename: str, model_dir: Path, logger: Optional[logging.Logger], use_logger: bool = True) -> None:
        file_path = model_dir / (filename + '.model')
        with open(file_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self._online_state))
        if use_logger:
            logger.info("Loaded model state from file: " + str(file_path))
    
    def load_model(self, filename: str, model_dir: Path, logger: Optional[logging.Logger], obs_shape: tuple, use_logger: bool = True) -> None:
        file_path = model_dir / filename
        template = TrainState.create(apply_fn=self._q_network.apply,
                                     params=self._q_network.init(jax.random.PRNGKey(201), jnp.empty(obs_shape, dtype=jnp.float32)),
                                     tx=optax.adam(learning_rate=0.0))
        with open(file_path, "rb") as f:
            self._online_state = flax.serialization.from_bytes(template, f.read())
        if use_logger:
            logger.info("Loaded model state from file: " + str(file_path))
