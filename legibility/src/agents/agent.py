#! /usr/bin/env python

import jax.numpy as jnp
import jax

from dl_algos.dqn import DQNetwork
from typing import Dict, List, Tuple
from logging import Logger


class Agent(object):
	
	_agent_id: int
	_goal_models: Dict[str, DQNetwork]
	_tasks: List[str]
	_n_tasks: int
	_rng_key: jax.random.PRNGKey
	
	def __init__(self, agent_id: int, goal_models: Dict[str, DQNetwork], rng_seed: int = 1234567890):
		self._agent_id = agent_id
		self._goal_models = goal_models
		self._tasks = []
		self._n_tasks = 0
		self._rng_key = jax.random.PRNGKey(rng_seed)
	
	@property
	def agent_id(self):
		return self._agent_id
	
	@property
	def goal_models(self) -> Dict[str, DQNetwork]:
		return self._goal_models
	
	@property
	def tasks(self) -> List[str]:
		return self._tasks
	
	@property
	def n_tasks(self) -> int:
		return self._n_tasks
	
	@n_tasks.setter
	def n_tasks(self, n_tasks: int) -> None:
		self._n_tasks = n_tasks
	
	@tasks.setter
	def tasks(self, tasks: List[str]) -> None:
		self._tasks = tasks.copy()
	
	@goal_models.setter
	def goal_models(self, new_models: Dict[str, DQNetwork]) -> None:
		self._goal_models = new_models
	
	def add_model(self, task: str, model: DQNetwork) -> None:
		self._goal_models[task] = model
	
	def remove_model(self, task: str) -> None:
		self._goal_models.pop(task)
	
	def init_interaction(self, interaction_tasks: List[str]):
		self._tasks = interaction_tasks.copy()
		self._n_tasks = len(interaction_tasks)
	
	def get_actions(self, task_id: str, obs: jnp.ndarray) -> int:
		q = jax.device_get(self._goal_models[task_id].q_network.apply(self._goal_models[task_id].online_state.params, obs)[0])
		pol = jnp.isclose(q, q.max(), rtol=1e-10, atol=1e-10).astype(int)
		pol = pol / pol.sum()
		# print(self._agent_id, task_id, q, q - q.max(), pol)
		
		self._rng_key, subkey = jax.random.split(self._rng_key)
		return int(jax.random.choice(subkey, len(q), p=pol))
	
	def action(self, obs: jnp.ndarray, sample: Tuple[jnp.ndarray, int], conf: float, logger: Logger, task: str = '') -> int:
		action = self.get_actions(task, obs)
		# print(self._agent_id, task, action)
		return action
	
	def sub_acting(self, obs: jnp.ndarray, logger: Logger, act_try: int, sample: Tuple[jnp.ndarray, int], conf: float, task: str = '') -> int:
		q_vals = jax.device_get(self._goal_models[task].q_network.apply(self._goal_models[task].online_state.params, obs)[0])
		sorted_q = jnp.copy(q_vals)
		sorted_q.sort()
		n_actions = len(sorted_q)
		
		self._rng_key, subkey = jax.random.split(self._rng_key)
		if act_try > n_actions:
			return int(jax.random.choice(subkey, n_actions))
		
		nth_best = sorted_q[max(-act_try, -n_actions)]
		return int(jax.random.choice(subkey, jnp.where(q_vals == nth_best)[0]))
