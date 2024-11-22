#! /usr/bin/env python

import jax.numpy as jnp
import jax

from dl_algos.dqn import DQNetwork
from agents.agent import Agent
from typing import Dict, List, Tuple
from logging import Logger


class TomAgent(Agent):
	
	_goal_prob: jnp.ndarray
	_sample_models: Dict[str, DQNetwork]
	_interaction_likelihoods: jnp.ndarray
	_sign: float
	_predict_task: str
	
	def __init__(self, agent_id: int, goal_models: Dict[str, DQNetwork], sample_models: Dict[str, DQNetwork], rng_seed: int = 1234567890, sign: float = -1):
		
		super().__init__(agent_id, goal_models, rng_seed)
		self._sample_models = sample_models
		self._goal_prob = jnp.array([])
		self._interaction_likelihoods = jnp.array([])
		self._sign = sign
		self._predict_task = ''
	
	@property
	def goal_prob(self) -> jnp.ndarray:
		return self._goal_prob
	
	@property
	def interaction_likelihoods(self) -> jnp.ndarray:
		return self._interaction_likelihoods
	
	@property
	def predict_task(self) -> str:
		return self._predict_task
	
	@property
	def sample_models(self) -> Dict[str, DQNetwork]:
		return self._sample_models
	
	@goal_prob.setter
	def goal_prob(self, goal_prob: jnp.ndarray) -> None:
		self._goal_prob = goal_prob
	
	@sample_models.setter
	def sample_models(self, sample_models: Dict[str, DQNetwork]) -> None:
		self._sample_models = sample_models
	
	def add_sample_model(self, task: str, model: DQNetwork) -> None:
		self._sample_models[task] = model
	
	def remove_sample_model(self, task: str) -> None:
		self._sample_models.pop(task)
	
	def init_interaction(self, interaction_tasks: List[str]):
		self._tasks = interaction_tasks.copy()
		self._n_tasks = len(interaction_tasks)
		self._goal_prob = jnp.ones(self._n_tasks) / self._n_tasks
		self._interaction_likelihoods = jnp.ones(self._n_tasks)
		self._predict_task = interaction_tasks[0]
	
	def reset_inference(self, tasks: List = None):
		if tasks:
			self._tasks = tasks.copy()
			self._n_tasks = len(self._tasks)
		self._interaction_likelihoods = jnp.ones(self._n_tasks)
		self._goal_prob = jnp.ones(self._n_tasks) / self._n_tasks
		self._predict_task = self._tasks[0]
	
	def sample_probability(self, obs: jnp.ndarray, a: int, conf: float) -> jnp.ndarray:
		goals_likelihood = []
		
		for task_id in self._tasks:
			q = jax.device_get(self._sample_models[task_id].q_network.apply(self._sample_models[task_id].online_state.params, obs)[0])
			goals_likelihood += [jnp.exp(self._sign * conf * (q[a] - q.max())) / jnp.sum(jnp.exp(self._sign * conf * (q - q.max())))]

		goals_likelihood = jnp.array(goals_likelihood)
		return goals_likelihood
	
	def task_inference(self, logger: Logger) -> str:
		if not self._tasks:
			logger.info('[ERROR]: List of possible tasks not defined!!')
			return ''
		
		if len(self._interaction_likelihoods) > 0:
			likelihood = jnp.cumprod(jnp.array(self._interaction_likelihoods), axis=0)[-1]
		else:
			likelihood = jnp.zeros(self._n_tasks)
		goals_prob = self._goal_prob * likelihood
		goals_prob_sum = goals_prob.sum()
		if goals_prob_sum == 0:
			p_max = jnp.ones(self._n_tasks) / self._n_tasks
		else:
			p_max = goals_prob / goals_prob_sum
		high_likelihood = jnp.argwhere(p_max == jnp.amax(p_max)).ravel()
		self._rng_key, subkey = jax.random.split(self._rng_key)
		return self._tasks[jax.random.choice(subkey, high_likelihood)]
	
	def bayesian_task_inference(self, sample: Tuple[jnp.ndarray, int], conf: float, logger: Logger) -> Tuple[str, float]:
		
		if not self._tasks:
			logger.info('[ERROR]: List of possible tasks not defined!!')
			return '', -1
		
		state, action = sample
		sample_prob = self.sample_probability(state, action, conf)
		self._interaction_likelihoods = jnp.vstack((self._interaction_likelihoods, sample_prob))
		
		likelihoods = jnp.cumprod(self._interaction_likelihoods, axis=0)[-1]
		goals_prob = likelihoods * self._goal_prob
		goals_prob_sum = goals_prob.sum()
		if goals_prob_sum == 0:
			p_max = jnp.ones(self._n_tasks) / self._n_tasks
		else:
			p_max = goals_prob / goals_prob_sum
		max_idx = jnp.argwhere(p_max == jnp.amax(p_max)).ravel()
		self._rng_key, subkey = jax.random.split(self._rng_key)
		max_task_prob = jax.random.choice(subkey, max_idx)
		task_conf = float(p_max[max_task_prob])
		task_id = self._tasks[max_task_prob]
		
		return task_id, task_conf
	
	def action(self, obs: jnp.ndarray, sample: Tuple[jnp.ndarray, int], conf: float, logger: Logger, task: str = '') -> int:
		if task == '':
			predict_task, predict_conf = self.bayesian_task_inference(sample, conf, logger)
			self._predict_task = predict_task
		return super().action(obs, sample, conf, logger, self._predict_task)
	
	def sub_acting(self, obs: jnp.ndarray, logger: Logger, act_try: int, sample: Tuple[jnp.ndarray, int], conf: float, task: str = '') -> int:
		if task == '':
			predict_task, predict_conf = self.bayesian_task_inference(sample, conf, logger)
			self._predict_task = predict_task
		return super().sub_acting(obs, logger, act_try, sample, conf, self._predict_task)


