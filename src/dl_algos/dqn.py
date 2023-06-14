#! /usr/bin/env python
import math
import pathlib
import random
import time
import os
from distutils.util import strtobool

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete, Space
from typing import Callable, List
from pathlib import Path
from termcolor import colored

EPS_TYPE = flax.core.FrozenDict({'linear': 1, 'exp': 2, 'log': 3, 'epoch': 4})


class QNetwork(nn.Module):
    action_dim: int
    num_layers: int
    layer_sizes: List[int]
    activation_function: Callable

    # def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int]):
    #     super().__init__()
    #     self._action_dim = action_dim
    #     self._num_layers = num_layers
    #     self._activation_function = act_function
    #     self._layer_sizes = layer_sizes.copy()

    @nn.compact
    def __call__(self, x_orig: jnp.ndarray):
        x = x_orig.copy()
        for i in range(self.num_layers):
            x = nn.Dense(self.layer_sizes[i])(x)
            x = self.activation_function(x)
        return nn.Dense(self.action_dim)(x)


class DQNetwork(object):

    _q_network: QNetwork
    _network_state: TrainState
    _target_params: flax.core.FrozenDict
    _replay_buffer: ReplayBuffer
    _tensorboard_writer: SummaryWriter
    _gamma: float
    
    def __init__(self, action_dim: int, num_layers: int, act_function: Callable, layer_sizes: List[int], buffer_size: int, gamma: float,
                 observation_space: Space, use_gpu: bool, handle_timeout: bool, use_tensorboard: bool = False, tensorboard_data: List = None):
    
        """
        Initializes a DQN
        
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
        
        self._q_network = QNetwork(action_dim=action_dim, num_layers=num_layers, activation_function=act_function, layer_sizes=layer_sizes.copy())
        self._replay_buffer = ReplayBuffer(buffer_size, observation_space, Discrete(action_dim), "cuda" if use_gpu else "cpu",
                                           handle_timeout_termination=handle_timeout)
        self._gamma = gamma
        self._use_summary = use_tensorboard
        self._target_params = None
        self._network_state = None
        if use_tensorboard:
            summary_log = tensorboard_data[0]
            queue_size = tensorboard_data[1]
            flush_time = tensorboard_data[2]
            file_suffix = tensorboard_data[3]
            self._tensorboard_writer = SummaryWriter(log_dir=summary_log, max_queue=queue_size, flush_secs=flush_time, filename_suffix=file_suffix)

    #############################
    ##    GETTERS & SETTERS    ##
    #############################

    @property
    def q_network(self) -> QNetwork:
        return self._q_network
    
    @property
    def network_state(self) -> TrainState:
        return self._network_state
    
    @property
    def target_params(self) -> flax.core.FrozenDict:
        return self._target_params
    
    @property
    def replay_buffer(self) -> ReplayBuffer:
        return self._replay_buffer
    
    @property
    def summary_writer(self) -> SummaryWriter:
        return self._tensorboard_writer
    
    @property
    def gamma(self) -> float:
        return self._gamma
    
    @gamma.setter
    def gamma(self, new_gamma: float) -> None:
        self._gamma = new_gamma
        
    @target_params.setter
    def target_params(self, new_params: flax.core.FrozenDict) -> None:
        self._target_params = new_params
        
    @network_state.setter
    def network_state(self, new_state: TrainState) -> None:
        self._network_state = new_state

    #############################
    ##       CLASS UTILS       ##
    #############################

    @jax.jit
    def update(self, q_state: TrainState, observations: np.ndarray, actions: np.ndarray, next_observations: np.ndarray, rewards: np.ndarray, dones: np.ndarray):
        q_next_target = self._q_network.apply(self._target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * self._gamma * q_next_target
    
        def mse_loss(params):
            q = self._q_network.apply(params, observations)  # (batch_size, num_actions)
            q = q[np.arange(q.shape[0]), actions.squeeze()]  # (batch_size,)
            return ((q - next_q_value) ** 2).mean(), q
    
        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    def train(self, env: gym.Env, total_timesteps: int, batch_size: int, optim_learn_rate: float, tau: float, initial_eps: float,
              final_eps: float, eps_type: str, rng_seed: int, exploration_decay: float = 0.99, warmup: int = 0, target_freq: int = 100, train_freq: int = 1,
              summary_frequency: int = 1):
        
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

        obs = env.reset()
        if self._network_state is None:
            self._network_state = TrainState.create(
                apply_fn=self._q_network.apply,
                params=self._q_network.init(q_key, obs),
                tx=optax.adam(learning_rate=optim_learn_rate),
            )
        if self._target_params is None:
            self._target_params = self._q_network.init(q_key, obs)

        self._q_network.apply = jax.jit(self._q_network.apply)
        self._target_params = optax.incremental_update(self._network_state.params, self._target_params, 1.0)

        start_time = time.time()
        to_convergence = (total_timesteps < 0)
        done = False
        epoch = 0
        loss = math.inf
        
        while not done:
            
            # interact with environment
            eps = eps_update(EPS_TYPE[eps_type], initial_eps, final_eps, exploration_decay, epoch, total_timesteps)
            if random.random() < eps:
                actions = np.array([env.action_space.sample()])
            else:
                q_values = self._q_network.apply(self._network_state.params, obs)
                actions = q_values.argmax(axis=-1)
                actions = jax.device_get(actions)
            next_obs, rewards, dones, infos = env.step(actions)
            
            if self._use_summary:
                for info in infos:
                    if "episode" in info.keys():
                        print(f"global_step={epoch}, episodic_return={info['episode']['r']}")
                        self._tensorboard_writer.add_scalar("charts/episodic_return", info["episode"]["r"], epoch)
                        self._tensorboard_writer.add_scalar("charts/episodic_length", info["episode"]["l"], epoch)
                        self._tensorboard_writer.add_scalar("charts/epsilon", eps, epoch)
                        break
            
            # store new samples
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d:
                    real_next_obs[idx] = infos[idx]["terminal_observation"]
            self._replay_buffer.add(obs, real_next_obs, actions, rewards, dones, infos)
            obs = next_obs
    
            # update Q-network and target network
            if epoch > warmup:
                if epoch % train_freq == 0:
                    data = self._replay_buffer.sample(batch_size)
                    # perform a gradient-descent step
                    loss, old_val, self._network_state = self.update(
                        self._network_state,
                        data.observations.numpy(),
                        data.actions.numpy(),
                        data.next_observations.numpy(),
                        data.rewards.flatten().numpy(),
                        data.dones.flatten().numpy(),
                    )
                    
                    if self._use_summary and epoch % summary_frequency == 0:
                        self._tensorboard_writer.add_scalar("losses/td_loss", jax.device_get(loss), epoch)
                        self._tensorboard_writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), epoch)
                        print("SPS:", int(epoch / (time.time() - start_time)))
                        self._tensorboard_writer.add_scalar("charts/SPS", int(epoch / (time.time() - start_time)), epoch)
                    
                if epoch % target_freq == 0:
                    self._target_params = optax.incremental_update(self._network_state.params, self._target_params, tau)

                if to_convergence and abs(loss) < 1e-9 or not to_convergence and epoch >= total_timesteps:
                    done = True
                    
            epoch += 1
            
    def get_action(self, obs):
        q_values = self._q_network.apply(self._q_network.variables, obs)
        actions = q_values.argmax(axis=-1)
        return jax.device_get(actions)
    
    def create_checkpoint(self, model_dir: Path, epoch: int = 0) -> None:
        save_checkpoint(ckpt_dir=model_dir, target=self._network_state, step=epoch)
    
    def load_checkpoint(self, ckpt_file: Path, epoch: int = -1) -> None:
        template = TrainState.create(apply_fn=self._q_network.apply,
                                     params=self._q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7)), train=False),
                                     tx=optax.adam(learning_rate=0.0001))
        if epoch < 0:
            if pathlib.Path.is_file(ckpt_file):
                self._network_state = restore_checkpoint(ckpt_dir=ckpt_file, target=template)
            else:
                print(colored('ERROR!! Could not load checkpoint, expected checkpoint file got directory instead', 'red'))
        else:
            if pathlib.Path.is_dir(ckpt_file):
                self._network_state = restore_checkpoint(ckpt_dir=ckpt_file, target=template, step=epoch)
            else:
                print(colored('ERROR!! Could not load checkpoint, expected checkpoint directory got file instead', 'red'))
    
    def save_model(self, filename: str, model_dir: Path) -> None:
        file_path = model_dir / (filename + '.model')
        with open(file_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self._network_state))
        print("Model state saved to file: " + str(file_path))
    
    def load_model(self, filename: str, model_dir: Path) -> None:
        file_path = model_dir / filename
        template = TrainState.create(apply_fn=self._q_network.apply,
                                     params=self._q_network.init(jax.random.PRNGKey(201), jnp.empty((1, 7)), train=False),
                                     tx=optax.adam(learning_rate=0.0001))
        with open(file_path, "rb") as f:
            self._network_state = flax.serialization.from_bytes(template, f.read())
        print("Loaded model state from file: " + str(file_path))
