#! /usr/bin/env python

import numpy as np
import argparse
import yaml
import jax
import logging
import os
import multiprocessing as mp
import csv

from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Callable
from gymnasium.spaces import MultiBinary
from agents.agent import Agent
from dl_envs.pursuit.pursuit_env import TargetPursuitEnv, Action, ActionDirection
from dl_algos.dqn import DQNetwork
from flax.linen import relu

LEADER_ID = 0
TOM_ID = 1
RNG_SEED = 20240729
CONF = 1.0
PREY_TYPES = {'idle': 0, 'greedy': 1, 'random': 2}

# ! /usr/bin/env python

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
		model_id = list(self._goal_models.keys())[0]
		
		for task_idx in range(self._n_tasks):
			q = jax.device_get(self._sample_models[model_id].q_network.apply(self._sample_models[model_id].online_state.params, obs[task_idx])[0])
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
		
		states, action = sample
		sample_prob = self.sample_probability(states, action, conf)
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
	
	def get_actions(self, task_id: str, obs: jnp.ndarray) -> int:
		model_id = list(self._goal_models.keys())[0]
		q = jax.device_get(self._goal_models[model_id].q_network.apply(self._goal_models[model_id].online_state.params, obs)[0])
		pol = jnp.isclose(q, q.max(), rtol=1e-10, atol=1e-10).astype(int)
		pol = pol / pol.sum()
		# print(self._agent_id, task_id, q, q - q.max(), pol)
		
		self._rng_key, subkey = jax.random.split(self._rng_key)
		return int(jax.random.choice(subkey, len(q), p=pol))
	
	def action(self, obs: jnp.ndarray, sample: Tuple[jnp.ndarray, int], conf: float, logger: Logger, task: str = '') -> int:
		predict_task, predict_conf = self.bayesian_task_inference(sample, conf, logger)
		self._predict_task = predict_task
		return self.get_actions(self._predict_task, obs[self._tasks.index(self._predict_task)])
	
	def sub_acting(self, obs: jnp.ndarray, logger: Logger, act_try: int, sample: Tuple[jnp.ndarray, int], conf: float, task: str = '') -> int:
		predict_task, predict_conf = self.bayesian_task_inference(sample, conf, logger)
		self._predict_task = predict_task
		return super().sub_acting(obs[self._tasks.index(self._predict_task)], logger, act_try, sample, conf, self._predict_task if task == '' else task)


def write_results_file(data_dir: Path, filename: str, results: Dict, logger: logging.Logger) -> None:
	try:
		with open(data_dir / (filename + '.csv'), 'w') as results_file:
			headers = ['test_nr', 'num_steps', 'pred_steps', 'average_pred_steps', 'num_caught_foods', 'food_steps', 'num_deadlocks']
			writer = csv.DictWriter(results_file, fieldnames=headers, delimiter=',', lineterminator='\n')
			writer.writeheader()
			for key in results.keys():
				row = {'test_nr': key}
				for header, val in zip(headers[1:], list(results[key])):
					row[header] = results[key][val]
				writer.writerow(row)

	except IOError as e:
		logger.error("I/O error: " + str(e))


def append_results_file(data_dir: Path, filename: str, results: Dict, logger: logging.Logger, test_nr: int) -> None:
	try:
		with open(data_dir / (filename + '.csv'), 'a') as results_file:
			headers = ['test_nr', 'num_steps', 'pred_steps', 'average_pred_steps', 'num_caught_foods', 'food_steps', 'num_deadlocks']
			writer = csv.DictWriter(results_file, fieldnames=headers, delimiter=',', lineterminator='\n')
			row = {'test_nr': test_nr}
			for header, key in zip(headers[1:], list(results.keys())):
				row[header] = results[key]
			writer.writerow(row)

	except IOError as e:
		logger.error("I/O error: " + str(e))


def is_deadlock(history: List, new_state: str, last_actions: Tuple) -> bool:
	
	if len(history) < 3:
		return False
	
	deadlock = True
	# if all([act == Action.NONE for act in last_actions]) or all([act == Action.LOAD for act in last_actions]):
	# 	return False
	#
	# else:
	state_repitition = 0
	for state in history:
		if new_state == state:
			state_repitition += 1
	if state_repitition < 3:
		deadlock = False
	
	return deadlock


def coordinate_agents(env: TargetPursuitEnv, predict_task: str, actions: Tuple[int], n_tom_agents: int) -> Tuple[int]:
	
	objective = env.target
	hunter_pos = [env.agents[h_id].pos for h_id in env.hunter_ids]
	objective_adj = env.adj_pos(env.agents[objective].pos)
	
	if sum([pos in objective_adj for pos in hunter_pos]) >= env.n_catch:
		if predict_task == str(objective):
			return tuple([Action.STAY] * env.n_hunters)
		else:
			return actions
	
	else:
		leader_pos = hunter_pos[LEADER_ID]
		lead_direction = ActionDirection[Action(actions[LEADER_ID]).name].value
		next_lead_pos = (leader_pos[0] + lead_direction[0], leader_pos[1] + lead_direction[1])
		tom_pos = []
		tom_directions = []
		next_tom_pos = []
		for idx in range(n_tom_agents):
			tom_pos.append(hunter_pos[TOM_ID + idx])
			tom_directions.append(ActionDirection[Action(actions[TOM_ID + idx]).name].value)
			next_tom_pos.append((tom_pos[-1][0] + tom_directions[-1][0], tom_pos[-1][1] + tom_directions[-1][1]))
			
		coord_acts = actions
		for idx in range(n_tom_agents):
			if next_tom_pos[idx] == next_lead_pos or (idx > 0 and next_tom_pos[idx] == next_tom_pos[idx - 1]):
				coord_acts = coord_acts[:TOM_ID + idx] + (Action.STAY.value, ) + coord_acts[TOM_ID + idx + 1:]
				
		return coord_acts
	

def load_models(logger: logging.Logger, opt_models_dir: Path, leg_models_dir: Path, n_hunters: int, prey_type: str, n_preys_alive: int, num_layers: int, act_function: Callable,
                layer_sizes: List[int], gamma: float, use_cnn: bool, use_dueling: bool, use_ddqn: bool, cnn_shape: Tuple, cnn_properties: List = None) -> Tuple[Dict, Dict]:
	optim_models = {}
	leg_models = {}
	opt_model_names = [fname.name for fname in (opt_models_dir / ('%d-hunters' % n_hunters) / ('%s-prey' % prey_type) / 'best').iterdir()]
	leg_model_names = [fname.name for fname in (leg_models_dir / ('%d-hunters' % n_hunters) / ('%s-prey' % prey_type) / 'best').iterdir()]
	try:
		# Find the optimal model name for the food location
		model_name = ''
		for name in opt_model_names:
			if name.find("%d" % n_preys_alive) != -1:
				model_name = name
				break
		assert model_name != ''
		opt_dqn = DQNetwork(len(Action), num_layers, act_function, layer_sizes, gamma, use_dueling, use_ddqn, use_cnn, cnn_properties)
		opt_dqn.load_model(model_name, opt_models_dir / ('%d-hunters' % n_hunters) / ('%s-prey' % prey_type)  / 'best', logger, cnn_shape, True)
		optim_models['p%d' % n_preys_alive] = opt_dqn
		
		# Find the legible model name for the food location
		model_name = ''
		for name in leg_model_names:
			if name.find("%d" % n_preys_alive) != -1:
				model_name = name
				break
		assert model_name != ''
		leg_dqn = DQNetwork(len(Action), num_layers, act_function, layer_sizes, gamma, use_dueling, use_ddqn, use_cnn, cnn_properties)
		leg_dqn.load_model(model_name, leg_models_dir / ('%d-hunters' % n_hunters) / ('%s-prey' % prey_type) / 'best', logger, cnn_shape, True)
		leg_models['p%d' % n_preys_alive] = leg_dqn
		
		return optim_models, leg_models
	
	except AssertionError as e:
		logger.error(e)
		return {}, {}


def run_test_iteration(start_optim_models: Dict, start_leg_models: Dict, logger: logging.Logger, test_mode: int, run_n: int, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]],
					   field_dims: Tuple[int, int], player_sight: int, prey_ids: List[str], prey_type: str, require_catch: bool, catch_reward: float, max_steps: int, rng_seed: int,
					   use_render: bool, use_cnn: bool, opt_models_dir: Path, leg_models_dir: Path, gamma: float, num_layers: int, act_function: Callable, layer_sizes: List[int],
                       use_dueling: bool, use_ddqn: bool, cnn_properties: List = None) -> Dict:
	
	# Initialize the agents for the interaction
	n_hunters = len(hunters)
	n_tom_hunters = n_hunters - 1
	n_preys = len(preys)
	if test_mode == 0:
		leader_agent = Agent(LEADER_ID, start_optim_models, rng_seed)
		tom_agents = [TomAgent(TOM_ID + idx, start_optim_models, start_optim_models, rng_seed, 1) for idx in range(n_tom_hunters)]
	elif test_mode == 1:
		leader_agent = Agent(LEADER_ID, start_optim_models, rng_seed)
		tom_agents = [TomAgent(TOM_ID + idx, start_leg_models, start_optim_models, rng_seed, 1) for idx in range(n_tom_hunters)]
	elif test_mode == 2:
		leader_agent = Agent(LEADER_ID, start_leg_models, rng_seed)
		tom_agents = [TomAgent(TOM_ID + idx, start_optim_models, start_leg_models, rng_seed, 1) for idx in range(n_tom_hunters)]
	else:
		leader_agent = Agent(LEADER_ID, start_leg_models, rng_seed)
		tom_agents = [TomAgent(TOM_ID + idx, start_leg_models, start_leg_models, rng_seed, 1) for idx in range(n_tom_hunters)]
	
	env = TargetPursuitEnv(hunters, preys, field_dims, player_sight, prey_ids[0], require_catch, max_steps, use_layer_obs=True, agent_centered=True, catch_reward=catch_reward)
	env.seed(rng_seed)
	it_results = {}
	rng_gen = np.random.default_rng(rng_seed)
	
	# Setup agents for test
	preys_left = prey_ids.copy()
	task = preys_left.pop(rng_gen.integers(n_preys))
	tasks = prey_ids.copy()
	tasks.sort()
	leader_agent.init_interaction(tasks)
	for idx in range(n_tom_hunters):
		tom_agents[idx].init_interaction(tasks)
	
	# Setup environment for test
	env.reset_init_pos()
	env.target = task
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	n_preys_alive = n_preys
	obs, *_ = env.reset()
	
	recent_states = [''.join([''.join(str(x) for x in env.agents[a_id].pos) for a_id in env.agents.keys() if env.agents[a_id].alive])]
	if use_cnn:
		leader_obs = obs[0].reshape((1, *cnn_shape))
		leader_sample = [env.make_target_grid_obs(prey)[LEADER_ID].reshape((1, *cnn_shape))  for prey in env.prey_alive_ids]
		tom_obs = [[env.make_target_grid_obs(prey)[idx].reshape((1, *cnn_shape))  for prey in env.prey_alive_ids] for idx in range(n_tom_hunters)]
	else:
		leader_obs = obs[0]
		leader_sample = [env.make_target_grid_obs(prey)[LEADER_ID] for prey in env.prey_alive_ids]
		tom_obs = [[env.make_target_grid_obs(prey)[idx] for prey in env.prey_alive_ids] for idx in range(n_tom_hunters)]
	actions = (leader_agent.action(leader_obs, (leader_sample, Action.STAY), CONF, logger, 'p%d' % n_preys_alive),
	           *[tom_agents[idx].action(tom_obs[idx], (leader_sample, Action.STAY), CONF, logger, 'p%d' % n_preys_alive) for idx in range(n_tom_hunters)])
	
	timeout = False
	n_steps = 0
	n_pred_steps = []
	steps_capture = []
	deadlock_states = []
	n_deadlocks = 0
	act_try = 0
	later_error = 0
	later_food_step = 0
	
	if use_render:
		env.render()
	
	logger.info('Started run number %d:' % (run_n + 1))
	logger.info(env.get_full_env_log())
	while n_preys_alive > 1 and not timeout:
		predicted_objectives = ','.join(['%s for tom agent %d' % (tom_agents[idx].predict_task, tom_agents[idx].agent_id) for idx in range(n_tom_hunters)])
		logger.info('Run number %d, step %d: remaining %d foods, predicted objective %s and real objective %s from ' % (run_n + 1, n_steps + 1, env.n_preys_alive,
																														predicted_objectives, task) + ', '.join(env.prey_alive_ids))
		n_steps += 1
		if use_cnn:
			last_leader_sample = ([env.make_target_grid_obs(prey)[LEADER_ID].reshape((1, *cnn_shape)) for prey in env.prey_alive_ids], actions[LEADER_ID])
		else:
			last_leader_sample = ([env.make_target_grid_obs(prey)[LEADER_ID] for prey in env.prey_alive_ids], actions[LEADER_ID])
		if any([task != tom_agents[idx].predict_task for idx in range(n_tom_hunters)]):
			later_error = n_steps
		obs, _, _, timeout, _ = env.step(actions)
		if use_render:
			env.render()
		
		if use_cnn:
			leader_obs = obs[0].reshape((1, *cnn_shape))
			tom_obs = [[env.make_target_grid_obs(prey)[idx].reshape((1, *cnn_shape))  for prey in env.prey_alive_ids] for idx in range(n_tom_hunters)]
		else:
			leader_obs = obs[0]
			tom_obs = [[env.make_target_grid_obs(prey)[idx] for prey in env.prey_alive_ids] for idx in range(n_tom_hunters)]
		
		if timeout:
			n_pred_steps += [later_error - later_food_step]
			steps_capture += [n_steps - later_food_step]
			break
		
		elif env.n_preys_alive < n_preys_alive:
			n_preys_alive = env.n_preys_alive
			n_pred_steps += [later_error - later_food_step]
			steps_capture += [n_steps - later_food_step]
			later_food_step = n_steps
			later_error = n_steps
			
			if env.n_preys_alive > 0:
				# Update tasks remaining and samples
				tasks = env.prey_alive_ids.copy()
				tasks.sort()
				for idx in range(n_tom_hunters):
					tom_agents[idx].init_interaction(tasks)
				if use_cnn:
					last_leader_sample = ([env.make_target_grid_obs(prey)[LEADER_ID].reshape((1, *cnn_shape))  for prey in env.prey_alive_ids], Action.STAY)
				else:
					last_leader_sample = ([env.make_target_grid_obs(prey)[LEADER_ID] for prey in env.prey_alive_ids], Action.STAY)
				recent_states = []
				
				# Update decision models
				optim_models, leg_models = load_models(logger, opt_models_dir, leg_models_dir, n_hunters, prey_type, n_preys_alive, num_layers, act_function, layer_sizes,
				                                       gamma, use_cnn, use_dueling, use_ddqn, cnn_shape, cnn_properties)
				if test_mode == 0:
					leader_agent.goal_models = optim_models
					for idx in range(n_tom_hunters):
						tom_agents[idx].goal_models = optim_models
						tom_agents[idx].sample_models = optim_models
				elif test_mode == 1:
					leader_agent.goal_models = optim_models
					for idx in range(n_tom_hunters):
						tom_agents[idx].goal_models = leg_models
						tom_agents[idx].sample_models = optim_models
				elif test_mode == 2:
					leader_agent.goal_models = leg_models
					for idx in range(n_tom_hunters):
						tom_agents[idx].goal_models = optim_models
						tom_agents[idx].sample_models = leg_models
				else:
					leader_agent.goal_models = leg_models
					for idx in range(n_tom_hunters):
						tom_agents[idx].goal_models = leg_models
						tom_agents[idx].sample_models = leg_models
				
				# Get next objective
				task = preys_left.pop(rng_gen.integers(n_preys_alive))
				env.target = task
		
		current_state = ''.join([''.join(str(x) for x in env.agents[a_id].pos) for a_id in env.agents.keys() if env.agents[a_id].alive])
		if is_deadlock(recent_states, current_state, actions):
			n_deadlocks += 1
			if current_state not in deadlock_states:
				deadlock_states.append(current_state)
			act_try += 1
			actions = (leader_agent.sub_acting(leader_obs, logger, act_try - 1, last_leader_sample, CONF, 'p%d' % n_preys_alive),
			           *[tom_agents[idx].sub_acting(tom_obs[idx], logger, act_try, last_leader_sample, CONF, 'p%d' % n_preys_alive)for idx in range(n_tom_hunters)])
			# actions = (leader_agent.action(leader_obs, last_leader_sample, CONF, logger, task), tom_agent.sub_acting(tom_obs, logger, act_try, last_leader_sample, CONF))
		else:
			act_try = 0
			actions = (leader_agent.action(leader_obs, last_leader_sample, CONF, logger, 'p%d' % n_preys_alive),
					   *[tom_agents[idx].action(tom_obs[idx], last_leader_sample, CONF, logger, 'p%d' % n_preys_alive) for idx in range(n_tom_hunters)])
		
		actions = coordinate_agents(env, [tom_agents[idx].predict_task for idx in range(n_tom_hunters)], actions, n_tom_hunters)
		
		recent_states.append(current_state)
		if len(recent_states) > 3:
			recent_states.pop(0)
	
	env.close()
	logger.info('Run Over!!')
	it_results['n_steps'] = n_steps
	it_results['pred_steps'] = n_pred_steps
	it_results['avg_pred_steps'] = np.mean(n_pred_steps) if len(n_pred_steps) > 0 else 0
	it_results['preys_captured'] = n_preys - n_preys_alive
	it_results['steps_capture'] = steps_capture
	it_results['deadlocks'] = n_deadlocks
	
	return it_results


def eval_legibility(n_runs: int, test_mode: int, logger: logging.Logger, opt_models_dir: Path, leg_models_dir: Path, field_dims: Tuple[int, int], hunters: List[Tuple[str, int]],
                    preys: List[Tuple[str, int]], player_sight: int, prey_ids: List[str], prey_type: str, require_catch: bool, catch_reward: float, max_steps: int, gamma: float,
                    num_layers: int, act_function: Callable, layer_sizes: List[int], use_cnn: bool, use_dueling: bool, use_ddqn: bool, data_dir: Path,
                    cnn_properties: List = None, run_paralell: bool = False, use_render: bool = False, start_run: int = 0):
	
	env = TargetPursuitEnv(hunters, preys, field_dims, player_sight, prey_ids[0], require_catch, max_steps, use_layer_obs=True, agent_centered=True, catch_reward=catch_reward)
	if isinstance(env.observation_space, MultiBinary):
		obs_space = MultiBinary([*env.observation_space.shape[1:]])
	else:
		obs_space = env.observation_space[0]
	cnn_shape = (0,) if not use_cnn else (*obs_space.shape[1:], obs_space.shape[0])
	
	optim_models, leg_models = load_models(logger, opt_models_dir, leg_models_dir, env.n_hunters, prey_type, env.n_preys, num_layers, act_function, layer_sizes, gamma, use_cnn,
	                                       use_dueling, use_ddqn, cnn_shape, cnn_properties)
	
	if run_paralell:
		results = {}
		t_pool = mp.Pool(int(0.75 * mp.cpu_count()))
		pool_results = [t_pool.apply_async(run_test_iteration, args=(optim_models, leg_models, logger, test_mode, run, hunters, preys, field_dims, player_sight, prey_ids, prey_type,
																	 require_catch, catch_reward, max_steps, RNG_SEED + run, use_render, use_cnn, opt_models_dir, leg_models_dir, gamma,
																	 num_layers, act_function, layer_sizes, use_dueling, use_ddqn, cnn_properties)) for run in range(start_run, n_runs)]
		t_pool.close()
		for idx in range(len(pool_results)):
			results[idx] = list(pool_results[idx].get())
		t_pool.join()

		if start_run < 1:
			write_results_file(data_dir / 'performances' / 'pursuit',
			                   'test_mode-%d_field_%d-%d_hunters-%d_%s-prey' % (test_mode, field_dims[0], field_dims[1], len(hunters), prey_type), results, logger)

		else:
			append_results_file(data_dir / 'performances' / 'pursuit',
			                   'test_mode-%d_field_%d-%d_hunters-%d_%s-prey' % (test_mode, field_dims[0], field_dims[1], len(hunters), prey_type), results, logger, start_run)

	else:
		for run in range(start_run, n_runs):
			results = run_test_iteration(optim_models, leg_models, logger, test_mode, run, hunters, preys, field_dims, player_sight, prey_ids, prey_type,
										 require_catch, catch_reward, max_steps, RNG_SEED + run, use_render, use_cnn, opt_models_dir, leg_models_dir, gamma,
										 num_layers, act_function, layer_sizes, use_dueling, use_ddqn, cnn_properties)

			if run < 1:
				write_results_file(data_dir / 'performances' / 'pursuit',
				                   'test_mode-%d_field_%d-%d_hunters-%d_%s-prey' % (test_mode, field_dims[0], field_dims[1], len(hunters), prey_type), {run: results}, logger)

			else:
				append_results_file(data_dir / 'performances' / 'pursuit',
				                    'test_mode-%d_field_%d-%d_hunters-%d_%s-prey' % (test_mode, field_dims[0], field_dims[1], len(hunters), prey_type), results, logger, run)

			logger.info('Run %d results: ' % run + str(results))

def main():
	
	parser = argparse.ArgumentParser(description='Testing level based foraging with different legible agent configurations.')
	
	# Test configuration
	parser.add_argument('--mode', dest='mode', type=int, required=True, choices=[0, 1, 2, 3],
	                    help='Team composition mode:'
	                         '\n\t0 - Optimal agent controls interaction with an optimal follower '
	                         '\n\t1 - Legible agent controls interaction with a legible follower '
	                         '\n\t2 - Legible agent controls interaction with an optimal follower'
	                         '\n\t3 - Optimal agent controls interaction with a legible follower')
	parser.add_argument('--runs', dest='nruns', type=int, required=True, help='Number of trial runs to obtain eval')
	parser.add_argument('--render', dest='render', action='store_true', help='Activate the render to see the interaction')
	parser.add_argument('--paralell', dest='paralell', action='store_true',
	                    help='Use paralell computing to speed the evaluation process. (Can\'t be used with render or gpu active)')
	parser.add_argument('--use_gpu', dest='gpu', action='store_true', help='Use gpu for matrix computations')
	parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
	                    help='Directory to store trained models and load optimal models, if left blank stored in default location')
	parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
	                    help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
	parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
	parser.add_argument('--fraction', dest='fraction', type=str, default='0.5', help='Fraction of JAX memory pre-compilation')
	parser.add_argument('--start-run', dest='start_run', type=int, default=0, help='Starting test run number')
	
	# Environment configuration
	parser.add_argument('--catch-reward', dest='catch_reward', type=float, required=False, default=5.0, help='Catch reward for catching a prey')
	parser.add_argument('--field-size', dest='field_lengths', type=int, nargs='+', required=True, help='Length and width of the field')
	parser.add_argument('--hunter-classes', dest='hunter_class', type=int, required=True, help='Class of agent to use for the hunters')
	parser.add_argument('--hunter-ids', dest='hunter_ids', type=str, nargs='+', required=True, help='List with the hunter ids in the environment')
	parser.add_argument('--n-hunters-catch', dest='require_catch', type=int, required=True, help='Minimum number of hunters required to catch a prey')
	parser.add_argument('--prey-ids', dest='prey_ids', type=str, nargs='+', required=True, help='List with the prey ids in the environment')
	parser.add_argument('--prey-type', dest='prey_type', type=str, required=True, choices=['idle', 'greedy', 'random'],
						help='Type of prey in the environment, possible types: idle, greedy or random')
	parser.add_argument('--steps-episode', dest='max_steps', type=int, required=True, help='Maximum number of steps an episode can to take')
	
	# Agent configuration
	parser.add_argument('--n-leg-agents', dest='n_leg_agents', type=int, default=1, help='Number of legible agents in the environment')
	parser.add_argument('--architecture', dest='architecture', type=str, required=True, help='DQN architecture to use from the architectures yaml')
	parser.add_argument('--gamma', dest='gamma', type=float, required=False, default=0.99, help='Discount factor for agent\'s future rewards')
	parser.add_argument('--cnn', dest='use_cnn', action='store_true', help='Flag that signals the use of a CNN as entry for the DQN architecture')
	parser.add_argument('--dueling', dest='dueling_dqn', action='store_true', help='Flag that signals the use of a Dueling DQN architecture')
	parser.add_argument('--ddqn', dest='use_ddqn', action='store_true', help='Flag that signals the use of a Double DQN')
	parser.add_argument('--vdn', dest='use_vdn', action='store_true', help='Flag that signals the use of a VDN DQN architecture')
	
	args = parser.parse_args()
	
	# Test parameter
	mode = args.mode
	n_runs = args.nruns
	use_render = args.render
	use_paralell = args.paralell
	use_gpu = args.gpu
	start_run = args.start_run
	
	# Environment parameters
	hunter_ids = args.hunter_ids
	prey_ids = args.prey_ids
	field_lengths = args.field_lengths
	steps_episode = args.max_steps
	require_catch = args.require_catch
	hunter_class = args.hunter_class
	prey_type = args.prey_type
	catch_reward = args.catch_reward
	
	# Agents parameters
	n_leg_agents = args.n_leg_agents
	architecture = args.architecture
	gamma = args.gamma
	use_cnn = args.use_cnn
	use_dueling_dqn = args.dueling_dqn
	use_ddqn = args.use_ddqn
	use_vdn = args.use_vdn
	
	os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = args.fraction
	if not use_gpu:
		jax.config.update('jax_platform_name', 'cpu')
	
	hunters = []
	preys = []
	n_hunters = len(hunter_ids)
	n_preys = len(prey_ids)
	for idx in range(n_hunters):
		hunters += [(hunter_ids[idx], hunter_class)]
	for idx in range(n_preys):
		preys += [(prey_ids[idx], PREY_TYPES[prey_type])]
	# print(gamma, initial_eps, final_eps, eps_decay, eps_type, warmup, learn_rate, target_learn_rate)
	field_dims = len(field_lengths)
	if 2 >= field_dims > 0:
		if field_dims == 1:
			field_size = (field_lengths[0], field_lengths[0])
			sight = field_lengths[0]
		else:
			field_size = (field_lengths[0], field_lengths[1])
			sight = max(field_lengths[0], field_lengths[1])
	else:
		logging.error('[ARGS ERROR] Field size must either be composed of only 1 or 2 arguments; %d were given. Exiting program' % field_dims)
		return
	
	now = datetime.now()
	home_dir = Path(__file__).parent.absolute().parent.absolute()
	log_dir = Path(args.logs_dir) if args.logs_dir != '' else home_dir / 'logs'
	data_dir = Path(args.data_dir) if args.data_dir != '' else home_dir / 'data'
	models_dir = Path(args.models_dir) if args.models_dir != '' else home_dir / 'models'
	log_filename = (('test_pursuit_coop_legibile_%dx%d-field_mode-%d_%d-hunters_%s-preys' % (field_size[0], field_size[1], mode, n_hunters, prey_type)) +
	                '_' + now.strftime("%Y%m%d-%H%M%S"))
	leg_models_dir = models_dir / ('pursuit_legible%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1]))
	opt_models_dir = models_dir / ('pursuit_single%s_dqn' % ('_vdn' if use_vdn else '')) / ('%dx%d-field' % (field_size[0], field_size[1]))
	
	with open(data_dir / 'configs' / 'q_network_architectures.yaml') as architecture_file:
		arch_data = yaml.safe_load(architecture_file)
		if architecture in arch_data.keys():
			n_layers = arch_data[architecture]['n_layers']
			layer_sizes = arch_data[architecture]['layer_sizes']
			n_conv_layers = arch_data[architecture]['n_cnn_layers']
			cnn_size = arch_data[architecture]['cnn_size']
			cnn_kernel = [tuple(elem) for elem in arch_data[architecture]['cnn_kernel']]
			pool_window = [tuple(elem) for elem in arch_data[architecture]['pool_window']]
			cnn_properties = [n_conv_layers, cnn_size, cnn_kernel, pool_window]
	
	if len(logging.root.handlers) > 0:
		for handler in logging.root.handlers:
			logging.root.removeHandler(handler)
	
	logger = logging.getLogger('pursuit_coop_legible_mode_%d_%s_preys' % (mode, prey_type))
	logger.setLevel(logging.INFO)
	file_handler = logging.FileHandler(log_dir / (log_filename + '.log'))
	file_handler.setFormatter(logging.Formatter('%(name)s %(asctime)s %(levelname)s:\t%(message)s'))
	file_handler.setLevel(logging.INFO)
	logger.addHandler(file_handler)
	
	eval_legibility(n_runs, mode, logger, opt_models_dir, leg_models_dir, field_size, hunters, preys, sight, prey_ids, prey_type, require_catch, catch_reward,
	                steps_episode, gamma, n_layers, relu, layer_sizes, use_cnn, use_dueling_dqn, use_ddqn, data_dir, cnn_properties, use_paralell, use_render, start_run)
	
	return 0


if __name__ == '__main__':
	main()
