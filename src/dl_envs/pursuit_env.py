#! /usr/bin/env python

import gymnasium
import numpy as np

from gymnasium import Env
from typing import Tuple, List, Dict, Callable, Any, TypeVar, SupportsFloat
from enum import IntEnum
from collections import defaultdict


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

# Reward constants
MOVE_REWARD = -0.2
CATCH_REWARD = 5
TOUCH_REWARD = 1
CATCH_ALL_REWARD = 10
EVADE_REWARD = 10
CAUGHT_REWARD = -10


class Action(IntEnum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	STAY = 4


class AgentType(IntEnum):
	HUNTER = 1
	PREY = 2


class Agent(object):
	
	_id: str
	_rank: int
	_agent_type: AgentType
	_pos: Tuple[int, int]
	_alive: bool
	
	def __init__(self, a_id: str, agent_type: int, rank: int = 0):
		
		self._id = a_id
		self._agent_type = agent_type
		self._pos = (-1, -1)
		self._rank = rank
		self._alive = True
		
	@property
	def agent_id(self) -> str:
		return self._id
	
	@property
	def pos(self) -> Tuple[int, int]:
		return self._pos
	
	@property
	def agent_type(self) -> int:
		return self._agent_type
	
	@property
	def rank(self) -> int:
		return self._rank
	
	@property
	def alive(self) -> bool:
		return self._alive
	
	@pos.setter
	def pos(self, new_pos: Tuple[int, int]) -> None:
		self._pos = new_pos
		
	@alive.setter
	def alive(self, new_alive: bool) -> None:
		self._alive = new_alive
	
	
class PursuitEnv(Env):
	
	_n_preys: int
	_n_hunters: int
	_field_size: Tuple[int, int]
	_hunter_ids: List[str]
	_prey_ids: List[str]
	_agents: Dict[str, Agent]
	_field: np.ndarray
	_hunter_sight: int
	_initial_pos: List[Dict[str, Tuple[int, int]]]
	_env_timestep: int
	_max_timesteps: int
	_n_catch: int
	_env_rng: np.random.default_rng
	
	def __init__(self, hunter_ids: List[str], prey_ids: List[str], field_size: Tuple[int, int], hunter_sight: int, n_catch: int = 4, max_steps: int = 250):
		
		self._hunter_ids = hunter_ids.copy()
		self._n_hunters = int(len(hunter_ids))
		self._prey_ids = prey_ids.copy()
		self._n_preys = int(len(prey_ids))
		self._field_size = field_size
		self._field = np.zeros(field_size)
		self._hunter_sight = hunter_sight
		self._initial_pos = [{} for _ in range(len(AgentType))]
		self._max_timesteps = max_steps
		self._n_catch = n_catch
		
		self._agents = {}
		rank = 1
		for h_id in hunter_ids:
			self._agents[h_id] = Agent(h_id, AgentType.HUNTER, rank)
			rank += 1
		
		rank = 0
		for p_id in prey_ids:
			self._agents[p_id] = Agent(p_id, AgentType.PREY, rank)
			rank += 1
			
	@property
	def n_preys(self) -> int:
		return self._n_preys
	
	@property
	def prey_ids(self) -> List[str]:
		return self._prey_ids
	
	@property
	def n_hunters(self) -> int:
		return self._n_hunters
	
	@property
	def hunter_ids(self) -> List[str]:
		return self._hunter_ids
	
	@property
	def agents(self) -> Dict[str, Agent]:
		return self._agents
	
	@property
	def field_size(self) -> Tuple[int, int]:
		return self._field_size
	
	@property
	def field(self) -> np.ndarray:
		return self._field
	
	@property
	def hunter_sight(self) -> int:
		return self._hunter_sight
	
	@property
	def env_timestep(self) -> int:
		return self._env_timestep
	
	@property
	def max_timesteps(self) -> int:
		return self._max_timesteps
	
	@property
	def init_pos(self) -> List:
		return self._initial_pos
	
	def seed(self, rng_seed: int) -> None:
		self._env_rng = np.random.default_rng(rng_seed)
	
	def reset_init_pos(self) -> None:
		self._initial_pos = [{} for _ in range(len(AgentType))]
	
	def adj_pos(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
		return list({(max(pos[0] - 1, 0), pos[1]), (min(pos[0] + 1, self._field_size[0] - 1), pos[1]),
					 (pos[0], max(pos[1] - 1, 0)), (pos[0], min(pos[1] + 1, self._field_size[1] - 1))})
	
	def spawn_hunters(self, init_pos: Dict[str, Tuple[int, int]] = None):
		
		rng_gen = self._env_rng
		if init_pos is None:
			hunter_initial_pos = self._initial_pos[AgentType.HUNTER - 1]
			initial_pos_id = list(hunter_initial_pos.keys())
			if len(hunter_initial_pos) > 0:
				for hunter in self._hunter_ids:
					hunter_idx = self._hunter_ids.index(hunter)
					if hunter_idx in initial_pos_id:
						self._agents[hunter].pos = hunter_initial_pos[hunter_idx]
						self._field[hunter_initial_pos[hunter_idx][0], hunter_initial_pos[hunter_idx][1]] = AgentType.HUNTER
					else:
						agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
						while self._field[agent_pos[0], agent_pos[1]] != 0:
							agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
						self._agents[hunter].pos = agent_pos
						self._initial_pos[AgentType.HUNTER - 1][self._hunter_ids.index(hunter)] = agent_pos
						self._field[agent_pos[0], agent_pos[1]] = AgentType.HUNTER
			else:
				for hunter in self._hunter_ids:
					agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
					while self._field[agent_pos[0], agent_pos[1]] != 0:
						agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
					self._agents[hunter].pos = agent_pos
					self._initial_pos[AgentType.HUNTER - 1][self._hunter_ids.index(hunter)] = agent_pos
					self._field[agent_pos[0], agent_pos[1]] = AgentType.HUNTER
		
		else:
			for hunter in self._hunter_ids:
				self._agents[hunter].pos = init_pos[hunter]
				self._initial_pos[AgentType.HUNTER - 1][self._hunter_ids.index(hunter)] = init_pos[hunter]
				self._field[init_pos[hunter][0], init_pos[hunter][1]] = AgentType.HUNTER
		
	def spawn_preys(self, init_pos: Dict[str, Tuple[int, int]] = None):
		
		rng_gen = self._env_rng
		if init_pos is None:
			prey_initial_pos = self._initial_pos[AgentType.PREY - 1]
			initial_pos_id = list(prey_initial_pos.keys())
			if len(prey_initial_pos) > 0:
				for prey in self._prey_ids:
					prey_idx = self._prey_ids.index(prey)
					if prey_idx in initial_pos_id:
						self._agents[prey].pos = prey_initial_pos[prey_idx]
						self._field[prey_initial_pos[prey_idx][0], prey_initial_pos[prey_idx][1]] = AgentType.PREY
					else:
						agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
						while self._field[agent_pos[0], agent_pos[1]] != 0:
							agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
						self._agents[prey].pos = agent_pos
						self._initial_pos[AgentType.PREY - 1][self._prey_ids.index(prey)] = agent_pos
						self._field[agent_pos[0], agent_pos[1]] = AgentType.PREY
			else:
				for prey in self._prey_ids:
					agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
					while self._field[agent_pos[0], agent_pos[1]] != 0:
						agent_pos = (rng_gen.choice(self._field_size[0]), rng_gen.choice(self._field_size[1]))
					self._agents[prey].pos = agent_pos
					self._initial_pos[AgentType.PREY - 1][self._prey_ids.index(prey)] = agent_pos
					self._field[agent_pos[0], agent_pos[1]] = AgentType.PREY
		
		else:
			for prey in self._prey_ids:
				self._agents[prey].pos = init_pos[prey]
				self._initial_pos[AgentType.PREY - 1][self._prey_ids.index(prey)] = init_pos[prey]
				self._field[init_pos[prey][0], init_pos[prey][1]] = AgentType.PREY
	
	def make_obs(self) -> np.ndarray:
		
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			agent_rank = self._agents[agent_id].rank
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				type_one_hot = [1, 0]
				hunter_obs += [agent_pos + type_one_hot + [agent_rank]]
			else:
				type_one_hot = [0, 1]
				prey_obs += agent_pos + type_one_hot + [agent_rank]
		
		final_obs = []
		for idx in range(len(hunter_obs)):
			ordered_hunter_obs = hunter_obs[idx].copy()
			for elem in hunter_obs[:idx]:
				ordered_hunter_obs += elem.copy()
			for elem in hunter_obs[idx+1:]:
				ordered_hunter_obs += elem.copy()
			final_obs += [ordered_hunter_obs + prey_obs]
		
		return np.array(final_obs)
	
	def get_rewards(self) -> np.ndarray:
		
		# If game ends and there are preys to be caught
		if self._env_timestep > self._env_timestep and self._n_preys > 0:
			rewards = [-EVADE_REWARD] * self._n_hunters
			for agent in self._agents:
				if self._agents[agent].agent_type == AgentType.PREY:
					rewards += [EVADE_REWARD]
			return np.array(rewards)
		
		# When all preys are caught, hunters get max reward and preys get max penalty
		if self._n_preys <= 1:
			is_surrounded = False
			for prey in self._prey_ids:
				prey_pos = self._agents[prey].pos
				prey_adj = self.adj_pos(prey_pos)
				is_surrounded = sum([self._field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj]) >= self._n_catch
			if is_surrounded:
				rewards = [CATCH_ALL_REWARD] * self._n_hunters
				for agent in self._agents:
					if self._agents[agent].agent_type == AgentType.PREY:
						rewards += [-CATCH_ALL_REWARD]
				return np.array(rewards)
		
		# By default, hunters get a reward for moving
		rewards_hunters = [MOVE_REWARD] * self._n_hunters
		rewards_preys = []
		all_hunter_ids = [agent_id for agent_id in self._agents.keys() if self._agents[agent_id].agent_type == AgentType.HUNTER]
		all_prey_ids = [agent_id for agent_id in self._agents.keys() if self._agents[agent_id].agent_type == AgentType.PREY]
		for prey in all_prey_ids:
			if prey in self._prey_ids:
				prey_pos = self._agents[prey].pos
				prey_adj = self.adj_pos(prey_pos)
				n_surround_hunters = sum([self._field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj])
				is_surrounded = n_surround_hunters >= self._n_catch
				# If prey is surrounded, gets a caught penalty and all hunters boxing it get a catching reward
				if is_surrounded:
					if self._n_preys == 1:
						for agent_id in all_hunter_ids:
							rewards_hunters[all_hunter_ids.index(agent_id)] = CATCH_ALL_REWARD
						rewards_preys += [-CATCH_ALL_REWARD]
					else:
						for agent_id in all_hunter_ids:
							if self._agents[agent_id].pos in prey_adj:
								rewards_hunters[all_hunter_ids.index(agent_id)] = CATCH_REWARD
						rewards_preys += [-CATCH_REWARD]
				# If prey is not surrounded, it gets a reward for being free and agents around it get a small reward
				else:
					rewards_preys += [-MOVE_REWARD]
					for agent_id in all_hunter_ids:
						if self._agents[agent_id].pos in prey_adj:
							rewards_hunters[all_hunter_ids.index(agent_id)] = n_surround_hunters * TOUCH_REWARD
			# If prey already caught, doesn't get reward
			else:
				rewards_preys += [0]
				
		return np.array(rewards_hunters + rewards_preys)
	
	def game_finished(self) -> bool:
		return self._n_preys <= 0 or self._env_timestep > self._max_timesteps
	
	def get_info(self) -> dict[str, Any]:
		if self._env_timestep > self._max_timesteps:
			return {'terminated': 'timeout'}
		else:
			return {}
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, np.ndarray, bool, dict[str, Any]]:
		
		n_agents = len(self._agents)
		self._env_timestep = 0
		
		# when current number of hunters and preys is less than intial, reset to initial state
		if (self._n_hunters + self._n_preys) < n_agents:
			n_preys = 0
			prey_ids = []
			for agent_id in self._agents.keys():
				agent_type = self._agents[agent_id].agent_type
				self._agents[agent_id].alive = True
				
				# reset preys in the field to the initial
				if agent_type == AgentType.PREY:
					prey_ids += [agent_id]
					n_preys += 1
			self._prey_ids = prey_ids
			self._n_preys = n_preys
		
		# reset field and agents positions
		self._field = np.zeros(self._field_size)
		self.spawn_hunters()
		self.spawn_preys()
		
		obs = self.make_obs()
		rewards = self.get_rewards()
		finished = self.game_finished()
		info = self.get_info()
		
		return obs, rewards, finished, info
	
	
	def step(self, actions: ActType) -> tuple[ObsType, np.ndarray, bool, dict[str, Any]]:
		
		# Attempt moving each agent
		collisions = {'hunter': defaultdict(list), 'prey': defaultdict(list)}
		for agent_id, agent_action in zip(self._agents, actions):
			if self._agents[agent_id].alive:																			# Verify agent is still in play
				agent_pos = self._agents[agent_id].pos
				agent_type = 'hunter' if agent_id.find('hunter') != -1 else 'prey'
				if agent_action == Action.UP:
					collisions[agent_type][max(agent_pos[0] - 1, 0), agent_pos[1]].append(agent_id)
				elif agent_action == Action.DOWN:
					collisions[agent_type][min(agent_pos[0] + 1, self._field_size[0] - 1), agent_pos[1]].append(agent_id)
				elif agent_action == Action.LEFT:
					collisions[agent_type][agent_pos[0], max(agent_pos[1] - 1, 0)].append(agent_id)
				elif agent_action == Action.RIGHT:
					collisions[agent_type][agent_pos[0], min(agent_pos[1] + 1, self._field_size[1] - 1)].append(agent_id)
				else:
					collisions[agent_type][agent_pos[0], agent_pos[1]].append(agent_id)
		
		# Resolve collisions for hunters
		colliding_pos = [key for key in collisions['hunter'].keys() if len(collisions['hunter'][key]) > 1]
		while len(colliding_pos) > 0:
			colliding_pos = self.resolve_collisions(colliding_pos, collisions['hunter'])
		
		# Update hunter positions
		for next_pos, ids in collisions['hunter'].items():
			agent_id = ids[0]
			curr_pos = self._agents[agent_id].pos
			if self._field[next_pos[0], next_pos[1]] == 0:
				self._agents[agent_id].pos = next_pos
		
		# Get rewards before updating captured preys
		rewards = self.get_rewards()
		
		# Check for captures
		captured_prey = []
		for prey_id in self._prey_ids:
			prey_adj = self.adj_pos(self._agents[prey_id].pos)
			is_surrounded = sum([self._field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj]) >= self._n_catch
			if is_surrounded:
				captured_prey += [prey_id]
				self._agents[prey_id].pos = (-1, -1)
				self._agents[prey_id].alive = False
		
		# Resolve collisions for preys
		colliding_pos = [key for key in collisions['prey'].keys() if len(collisions['prey'][key]) > 1]
		while len(colliding_pos) > 0:
			colliding_pos = self.resolve_collisions(colliding_pos, collisions['prey'])
		
		# Update prey positions
		for next_pos, ids in collisions['prey'].items():
			agent_id = ids[0]
			if self._agents[agent_id].alive:
				curr_pos = self._agents[agent_id].pos
				if self._field[next_pos[0], next_pos[1]] == 0:
					self._agents[agent_id].pos = next_pos
		
		# Update field
		new_field = np.zeros(self._field_size)
		for hunter in self._hunter_ids:
			hunter_pos = self._agents[hunter].pos
			new_field[hunter_pos[0], hunter_pos[1]] = AgentType.HUNTER
		for prey in self._prey_ids:
			prey_pos = self._agents[prey].pos
			new_field[prey_pos[0], prey_pos[1]] = AgentType.PREY
		self._field = new_field
		
		# Remove captured preys from play
		if len(captured_prey) > 0:
			for prey_id in captured_prey:
				self._prey_ids.remove(prey_id)
		self._n_preys = int(len(self._prey_ids))
		
		# Update timestep count
		self._env_timestep += 1
		return self.make_obs(), rewards, self.game_finished(), self.get_info()
	
	def resolve_collisions(self, colliding_pos: List, collisions: Dict) -> List:
		for pos in colliding_pos:
			colliding_agents = collisions[pos]
			pos_free = self._field[pos[0], pos[1]] == 0
			# Agents staying have priority over moving agents
			if not pos_free:
				for agent_id in colliding_agents:
					agent_pos = self._agents[agent_id].pos
					collisions[pos[0], pos[1]].remove(agent_id)
					collisions[agent_pos[0], agent_pos[1]].append(agent_id)
			# Agents with lower rank attribute have priority when moving
			else:
				low_rank = self._agents[colliding_agents[0]].rank
				moving_agent = colliding_agents[0]
				for idx in range(1, len(colliding_agents)):
					if self._agents[colliding_agents[idx]].rank < low_rank:
						low_rank = self._agents[colliding_agents[idx]].rank
						moving_agent = colliding_agents[idx]
				for agent_id in colliding_agents:
					if agent_id != moving_agent:
						agent_pos = self._agents[agent_id].pos
						collisions[agent_pos[0], agent_pos[1]].append(agent_id)
				collisions[pos] = [moving_agent]
		return [key for key in collisions.keys() if len(collisions[key]) > 1]
