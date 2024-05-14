#! /usr/bin/env python
import numpy as np
import jax

from gymnasium import spaces
from typing import Dict, Tuple, Union


def get_obs_shape(observation_space: spaces.Space) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]], Tuple[int, Dict]]:
    """
    Get the shape of the observation.

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box) or isinstance(observation_space, spaces.MultiDiscrete):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, spaces.MultiBinary):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]
    elif isinstance(observation_space, spaces.Tuple):
        n_obs = len(observation_space)
        obs_shape = get_obs_shape(observation_space[0])
        if isinstance(obs_shape, dict):
            return n_obs, obs_shape
        else:
            return n_obs, *obs_shape
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")
	

def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.
    For MultiDiscrete action spaces, assumes that actions are 1D arrays

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), "Multi-dimensional MultiBinary action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")
    

def get_device(device: Union[jax.Device, str] = "auto") -> jax.Device:
    """
    Retrieve tensor device.
    It checks that the requested device is available first.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu' or a jax.Device
    :return: Supported tensor device
    """
    
    # If already jax device return it
    if isinstance(device, jax.Device):
        return device
    
    # First GPU by default
    if device == "auto":
        device = jax.devices('gpu')[0]
    
    # Get device specified
    else:
        dev_splt = device.split(':')
        dev_desc = dev_splt[0]
        if len(dev_splt) == 1:
            device = jax.devices(dev_desc)[0]
        else:
            if dev_splt[1] != '':
                dev_idx = int(dev_splt[1])
                if dev_idx > jax.device_count(dev_desc):
                    device = jax.devices(dev_desc)[0]
                else:
                    device = jax.devices(dev_desc)[dev_idx]
            else:
                device = jax.devices(dev_splt[0])[0]
    
    # Cuda not available
    gpu_available = any(device.platform == 'gpu' for device in jax.devices())
    if not gpu_available:
        return jax.devices("cpu")[0]
    
    return device