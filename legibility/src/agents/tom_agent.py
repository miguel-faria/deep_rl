#! /usr/bin/env python

import jax.numpy as jnp
import jax

from dl_algos.dqn import DQNetwork
from agents.agent import Agent
from typing import Dict, List, Tuple
from logging import Logger


class TomAgent(Agent):

	_goal_prob: jnp.ndarray
	_interaction_likelihoods: List[float]
	_sign: float

	def __init__(self, goal_models: Dict[str, DQNetwork], rng_seed: int = 1234567890, sign: float = -1):

		super().__init__(goal_models, rng_seed)
		self._goal_prob = jnp.array([])
		self._interaction_likelihoods = []
		self._sign = sign

	@property
	def goal_prob(self) -> jnp.ndarray:
		return self._goal_prob

	@property
	def interaction_likelihoods(self) -> List[float]:
		return self._interaction_likelihoods

	@goal_prob.setter
	def goal_prob(self, goal_prob: jnp.ndarray) -> None:
		self._goal_prob = goal_prob

	def init_interaction(self, interaction_tasks: List[str]):
		self._tasks = interaction_tasks.copy()
		self._n_tasks = len(interaction_tasks)
		self._goal_prob = jnp.ones(self._n_tasks) / self._n_tasks

	def reset_inference(self, tasks: List = None):
		if tasks:
			self._tasks = tasks.copy()
			self._n_tasks = len(self._tasks)
		self._interaction_likelihoods = []
		self._goal_prob = jnp.ones(self._n_tasks) / self._n_tasks

	def sample_probability(self, obs: jnp.ndarray, a: int, conf: float) -> jnp.ndarray:
		goals_likelihood = []

		for task_id in self._tasks:
			q = jax.device_get(self._goal_models[task_id].q_network.apply(self._goal_models[task_id].online_state.params, obs))
			goals_likelihood += [jnp.exp(self._sign * conf * (q[a] - q.max())) / jnp.sum(jnp.exp(self._sign * conf * (q - q.max())))]

		goals_likelihood = jnp.array(goals_likelihood)
		return goals_likelihood

	def bayesian_task_inference(self, sample: Tuple[jnp.ndarray, int], conf: float, logger: Logger) -> Tuple[str, float]:

		if not self._tasks:
			logger.info('[ERROR]: List of possible tasks not defined!!')
			return '', -1

		state, action = sample
		sample_prob = self.sample_probability(state, action, conf)
		sample_likelihoods = self._goal_prob * sample_prob
		self._goal_prob += sample_prob
		self._goal_prob = self._goal_prob / self._goal_prob.sum()
		self._interaction_likelihoods += [sample_likelihoods]

		likelihoods = jnp.cumprod(jnp.array(self._interaction_likelihoods), axis=0)[-1]
		likelihood_sum = likelihoods.sum()
		if likelihood_sum == 0:
			p_max = jnp.ones(self._n_tasks) / self._n_tasks
		else:
			p_max = likelihoods / likelihood_sum
		max_idx = jnp.argwhere(p_max == jnp.amax(p_max)).ravel()
		self._rng_key, subkey = jax.random.split(self._rng_key)
		max_task_prob = jax.random.choice(subkey, max_idx)
		task_conf = float(p_max[max_task_prob])
		task_id = self._tasks[max_task_prob]

		return task_id, task_conf

	def action(self, obs: jnp.ndarray, sample: Tuple[jnp.ndarray, int], conf: float, logger: Logger, task: str = '') -> int:
		predict_task, _ = self.bayesian_task_inference(sample, conf, logger)
		return super().action(obs, sample, conf, logger, predict_task)

	def sub_acting(self, obs: jnp.ndarray, logger: Logger, act_try: int, sample: Tuple[jnp.ndarray, int], conf: float, task: str = '') -> int:
		predict_task, _ = self.bayesian_task_inference(sample, conf, logger)
		return super().sub_acting(obs, logger, act_try, sample, conf, predict_task)


