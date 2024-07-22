#! /usr/bin/env python
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Generator, Union

import gymnasium
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces

from dl_utilities.utilities import get_action_dim, get_obs_shape, get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    import psutil
except ImportError:
    psutil = None


class RolloutSample(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    old_values: jax.Array
    old_log_prob: jax.Array
    advantages: jax.Array
    returns: jax.Array


class ReplaySample(NamedTuple):
    observations: jax.Array
    actions: jax.Array
    next_observations: jax.Array
    dones: jax.Array
    rewards: jax.Array


class GeneralBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, device: Union[jax.Device, str] = "auto", n_envs: int = 1,
                 n_agents: int = 1, rng_seed: int = 1234567890):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = (n_agents, get_action_dim(action_space[0])) if isinstance(action_space, gymnasium.spaces.Tuple) else get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.start_seed = rng_seed
        self.rng_key = jax.random.PRNGKey(rng_seed)

    def reseed(self, rng_seed):
        self.start_seed = rng_seed
        self.rng_key = jax.random.PRNGKey(rng_seed)

    def reset_seed(self):
        self.rng_key = jax.random.PRNGKey(self.start_seed)
    
    def set_pos(self, new_pos: int) -> None:
        self.pos = min(new_pos, self.buffer_size)
        if new_pos < self.buffer_size:
            self.full = False
    
    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False
        self.reset_seed()

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, all_envs: bool = False):
        """
        Sample a batch of transitions from the buffer.
        
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :param all_envs: Whether to sample from all environments or just one
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        self.rng_key, subkey = jax.random.split(self.rng_key)
        batch_inds = jax.random.randint(subkey, shape=(batch_size, ), minval=0, maxval=upper_bound, dtype=int)
        return self._get_samples(batch_inds, env=env, all_envs=all_envs)

    def to_tensor(self, arr: Union[np.ndarray, jax.Array]) -> jax.Array:
        if not isinstance(arr, jax.Array):
            arr = jnp.array(arr)
        return jax.device_put(arr, device=self.device)

    @abstractmethod
    def _get_samples(self, batch_inds: Union[np.ndarray, jax.Array], env: Optional[VecNormalize] = None, all_envs: bool = False) -> Union[ReplaySample, RolloutSample]:
        """
        :param batch_inds:
        :param env:
        :param all_envs:
        :return:
        """
        raise NotImplementedError()

    @staticmethod
    def _normalize_obs(obs: Union[np.ndarray, Dict[str, np.ndarray]], env: Optional[VecNormalize] = None) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(GeneralBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, device: Union[jax.Device, str] = "auto", n_envs: int = 1, n_agents: int = 1,
                 rng_seed: int = 1234567890, optimize_memory_usage: bool = False, handle_timeout_termination: bool = True, smart_add: bool = False, add_method: str = 'uniform'):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, n_agents=n_agents,rng_seed=rng_seed)
        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)
        obs_shape = self.obs_shape

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.smart_add = smart_add
        self.add_method = add_method
        self.optimize_memory_usage = optimize_memory_usage
        self.samples_order = None
        self.pos_probs = None
        self.observations = np.zeros((self.buffer_size, self.n_envs, *obs_shape), dtype=observation_space.dtype)

        if optimize_memory_usage:
            self.next_observations = None
        else:
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
    
        if self.n_agents == 1:
            self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        else:
            self.rewards = np.zeros((self.buffer_size, self.n_envs, self.n_agents), dtype=np.float32)
        
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(self, obs: np.ndarray, next_obs: np.ndarray, action: Union[np.ndarray, int], reward: Union[np.ndarray, float], done: Union[np.ndarray, bool],
            infos: List[Dict[str, Any]]) -> None:
        if self.smart_add:
            if self.full:
                if self.add_method == 'weighted':
                    if self.samples_order is None:
                        self.samples_order = jnp.arange(self.buffer_size).tolist()
                    if self.pos_probs is None:
                        probs = self.buffer_size - jnp.arange(self.buffer_size)
                        self.pos_probs = probs / probs.sum()
                    _, subkey = jax.random.split(self.rng_key)
                    pop_pos = jax.random.choice(subkey, jnp.arange(self.buffer_size), p=self.pos_probs)
                    pos = self.samples_order.pop(pop_pos)
                    self.samples_order.append(pos)
                else:
                    _, subkey = jax.random.split(self.rng_key)
                    pos = jax.random.randint(subkey, minval=0, maxval=self.buffer_size, shape=())
                self.add_sample(pos, obs, next_obs, action, reward, done, infos)
            else:
                self.add_sample(self.pos, obs, next_obs, action, reward, done, infos)
                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
            
        else:
            self.add_sample(self.pos, obs, next_obs, action, reward, done, infos)
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True
                self.pos = 0

    def add_sample(self, pos: int, obs: np.ndarray, next_obs: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]):
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))
        self.observations[pos] = np.array(obs).copy()
        
        if self.optimize_memory_usage:
            self.observations[(pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[pos] = np.array(next_obs).copy()
        
        self.actions[pos] = np.array(action).copy()
        self.rewards[pos] = np.array(reward).copy()
        self.dones[pos] = np.array(done).copy()
        
        if self.handle_timeout_termination:
            self.timeouts[pos] = np.array([info.get("timeout", False) for info in infos])

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None, all_envs: bool = False) -> ReplaySample:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :param all_envs: Whether to sample from all environments or just one
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env, all_envs=all_envs)
        
        self.rng_key, subkey = jax.random.split(self.rng_key)
        if self.full:
            batch_inds = (jax.random.randint(subkey, shape=(batch_size, ), minval=0, maxval=self.buffer_size, dtype=int) + self.pos) % self.buffer_size
        else:
            batch_inds = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=self.pos, dtype=int)
        
        return self._get_samples(batch_inds, env=env, all_envs=all_envs)

    def _get_samples(self, batch_inds: Union[np.ndarray, jax.Array], env: Optional[VecNormalize] = None, all_envs: bool = False) -> ReplaySample:
        batch_size = len(batch_inds)
        if not all_envs:
            # Sample randomly the env idx
            if self.n_envs > 1:
                env_indices = jax.random.randint(self.rng_key, shape=(batch_size, ), minval=0, maxval=self.n_envs, dtype=int)
            else:
                env_indices = 0

            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(self.next_observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            else:
                next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
            
            rewards = (self.rewards[batch_inds].reshape(-1, 1) if self.n_agents < 2 else self.rewards[batch_inds].reshape(-1, *self.rewards.shape[2:]))
            dones = (self.dones[batch_inds] * (1 - self.timeouts[batch_inds])).reshape(-1, 1)
            
            data = (self._normalize_obs(self.observations[batch_inds, env_indices, :], env), self.actions[batch_inds, env_indices, :], next_obs,
                    dones, self._normalize_reward(rewards, env))
        
        else:
            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(self.next_observations[(batch_inds + 1) % self.buffer_size, :], env)
            else:
                next_obs = self._normalize_obs(self.next_observations[batch_inds, :], env)
    
            rewards_shape = self.rewards.shape
            rewards = (np.zeros((batch_size * rewards_shape[-1], self.n_envs, 1)) if self.n_agents < 2 else
                       np.zeros((batch_size, self.n_envs, *rewards_shape[2:])))
            for idx in range(0, self.n_envs):
                rewards[:, idx] = (self.rewards[batch_inds, idx].reshape(-1, 1) if self.n_agents < 2 else
                                   self.rewards[batch_inds, idx].reshape(-1, *rewards_shape[2:]))
            rewards = np.array(rewards)
    
            dones = np.zeros((batch_size, self.n_envs, 1))
            for idx in range(self.n_envs):
                dones[:, idx] = (self.dones[batch_inds, idx] * (1 - self.timeouts[batch_inds, 0])).reshape(-1, 1)
            dones = np.array(dones)
            
            data = (self._normalize_obs(self.observations[batch_inds, :], env), self.actions[batch_inds, :], next_obs, dones,
                    self._normalize_reward(rewards, env))
        
        return ReplaySample(*tuple(map(self.to_tensor, data)))


class RolloutBuffer(GeneralBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space, device: Union[jax.Device, str] = "auto",
                 gae_lambda: float = 1, gamma: float = 0.99, n_envs: int = 1, n_agents: int = 1):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs, n_agents=n_agents)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: jax.Array, dones: jnp.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to JAX array
        last_values = jnp.array(last_values)

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # TD(lambda) estimator
        self.returns = self.advantages + self.values

    def add(self, obs: np.ndarray, action: np.ndarray, reward: np.ndarray, episode_start: np.ndarray, value: jax.Array, log_prob: jax.Array) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = jnp.array(value)
        self.log_probs[self.pos] = jnp.array(log_prob)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutSample, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> RolloutSample:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutSample(*tuple(map(self.to_torch, data)))