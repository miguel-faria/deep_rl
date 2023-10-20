#! /usr/bin/env python

import numpy as np
import gymnasium

from gymnasium import Env
from typing import Tuple, List, Dict, Any, TypeVar
from enum import IntEnum, Enum
from collections import defaultdict
from termcolor import colored
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete, Space, Box
from dl_envs.pursuit.agents.target_agent import TargetAgent
from dl_envs.pursuit.agents.greedy_prey import GreedyPrey
from dl_envs.pursuit.agents.agent import Agent, AgentType

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

# Reward constants
MOVE_REWARD = 0.0
CATCH_REWARD = 5
TOUCH_REWARD = 0.025
CATCH_ALL_REWARD = 10
EVADE_REWARD = 10
CAUGHT_REWARD = -5


class ActionDirection(Enum):
	UP = (-1, 0)
	DOWN = (1, 0)
	LEFT = (0, -1)
	RIGHT = (0, 1)
	STAY = (0, 0)


class Action(IntEnum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	STAY = 4


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
	
	def __init__(self, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]], field_size: Tuple[int, int], hunter_sight: int, n_catch: int = 4,
				 max_steps: int = 250, use_encoding: bool = False, dead_preys: List[bool] = None, target_preys: List[str] = None, use_layer_obs: bool = False,
				 agent_centered: bool = False):
		
		self._hunter_ids = [hunter[0] for hunter in hunters]
		self._n_hunters = int(len(hunters))
		self._prey_ids = [prey[0] for prey in preys]
		self._n_preys = int(len(preys))
		self._field_size = field_size
		self._field = np.zeros(field_size)
		self._hunter_sight = hunter_sight
		self._initial_pos = [{} for _ in range(len(AgentType))]
		self._max_timesteps = max_steps
		self._n_catch = n_catch
		self._dead_preys = [dead_preys.copy() if dead_preys is not None else [False] * self._n_preys]
		self._target_preys = target_preys
		self._use_encoding = use_encoding
		self._center_agent = agent_centered
		self._use_layer_obs = use_layer_obs
		
		self._agents = {}
		rank = 1
		for h_id, h_type in hunters:
			if h_type == 0:
				self._agents[h_id] = Agent(h_id, AgentType.HUNTER, rank)
			else:
				self._agents[h_id] = TargetAgent(h_id, AgentType.HUNTER, rank)
		
		rank = 0
		for p_id, p_type in preys:
			if p_type == 0:
				self._agents[p_id] = Agent(p_id, AgentType.PREY, rank)
			else:
				self._agents[p_id] = GreedyPrey(p_id, AgentType.PREY, rank)
				
		self.action_space = gymnasium.spaces.Tuple(tuple([Discrete(len(Action))] * len(self._agents)))
		self.observation_space = gymnasium.spaces.Tuple(tuple([self._get_observation_space()] * self._n_hunters))
		
	###########################
	### GETTERS AND SETTERS ###
	###########################
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
	
	#######################
	### UTILITY METHODS ###
	#######################
	
	def _get_observation_space(self) -> Space:
		if self._use_layer_obs:
			# grid observation space
			grid_shape = (1 + 2 * self._hunter_sight, 1 + 2 * self._hunter_sight)
			
			# agents layer: agent levels
			hunters_min = np.zeros(grid_shape, dtype=np.int32)
			hunters_max = np.ones(grid_shape, dtype=np.int32)
			
			# foods layer: foods level
			preys_min = np.zeros(grid_shape, dtype=np.int32)
			preys_max = np.ones(grid_shape, dtype=np.int32)
			
			# access layer: i the cell available
			access_min = np.zeros(grid_shape, dtype=np.int32)
			access_max = np.ones(grid_shape, dtype=np.int32)
			
			# total layer
			min_obs = np.stack([hunters_min, preys_min, access_min])
			max_obs = np.stack([hunters_max, preys_max, access_max])
		
		else:
			if self._use_encoding:
				min_obs = [[-1, -1, 0, 0] * self._n_preys + [-1, -1, 0, 0] * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 1, 1] * self._n_preys +
						   [self._field_size[0], self._field_size[1], 1, 1] * self._n_hunters] * self._n_hunters
			else:
				min_obs = [[-1, -1, 2] * self._n_preys + [-1, -1, 1] * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 2] * self._n_preys +
						   [self._field_size[0], self._field_size[1], 1] * self._n_hunters] * self._n_hunters
			
		return Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)
	
	def sample_action(self) -> int:
		return self.action_space.sample()
	
	def set_target(self, hunter_id: str, prey_id: str) -> None:
		if isinstance(self._agents[hunter_id], TargetAgent):
			self._agents[hunter_id].target = prey_id
		else:
			print(colored('Agent class of agent %s does not support specific prey targets' % hunter_id, 'yellow'))
	
	def seed(self, seed=None):
		self._np_random, seed = seeding.np_random(seed)
		self.action_space.seed(seed)
		self.observation_space.seed(seed)
	
	def reset_init_pos(self) -> None:
		self._initial_pos = [{} for _ in range(len(AgentType))]
	
	def reset_timestep(self) -> None:
		self._env_timestep = 0
	
	def adj_pos(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
		return list({(max(pos[0] - 1, 0), pos[1]), (min(pos[0] + 1, self._field_size[0] - 1), pos[1]),
					 (pos[0], max(pos[1] - 1, 0)), (pos[0], min(pos[1] + 1, self._field_size[1] - 1))})
	
	def get_rewards(self, field: np.ndarray) -> np.ndarray:
		
		# If game ends and there are preys to be caught
		if self._env_timestep > self._env_timestep and self._n_preys > 0:
			rewards = [-EVADE_REWARD] * self._n_hunters
			for agent in self._agents:
				if self._agents[agent].agent_type == AgentType.PREY:
					rewards += [EVADE_REWARD]
			return np.array(rewards)
		
		# When all preys are caught, hunters get max reward and preys get max penalty
		if self._n_preys == 1:
			is_surrounded = False
			for prey in self._prey_ids:
				prey_pos = self._agents[prey].pos
				prey_adj = self.adj_pos(prey_pos)
				is_surrounded = sum([field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj]) >= self._n_catch
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
				n_surround_hunters = sum([field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj])
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
							rewards_hunters[all_hunter_ids.index(agent_id)] = n_surround_hunters * (CATCH_REWARD / self._n_catch)
			# If prey already caught, doesn't get reward
			else:
				rewards_preys += [0]
		
		return np.array(rewards_hunters + rewards_preys)
	
	def game_finished(self) -> bool:
		return self._n_preys <= 0
	
	def timeout(self) -> bool:
		return self._env_timestep > self._max_timesteps
	
	def get_info(self) -> dict[str, Any]:
		return {'preys_left': self._n_preys, 'timestep': self._env_timestep}
	
	def resolve_collisions(self, next_positions_d: Dict, prev_positions: List) -> None:
		
		next_positions = dict([(key, val) for at in next_positions_d.keys() for key, val in next_positions_d[at].items()])
		cant_move = {'hunter': [], 'prey': []}
		for a_id, a_pos in next_positions.items():
			add_move = True
			for oa_id, oa_pos in prev_positions:
				if a_id == oa_id:
					continue
				if ((a_pos == next_positions[oa_id] and not (self._agents[a_id].agent_type == AgentType.HUNTER and
															 self._agents[oa_id].agent_type == AgentType.PREY))
						or (a_pos == oa_pos and self._agents[a_id].pos == next_positions[oa_id])
						or self._field[a_pos[0], a_pos[1]] != 0):
					add_move = False
					break
			if not add_move:
				cant_move['hunter' if self._agents[a_id].agent_type == AgentType.HUNTER else 'prey'].append(a_id)
		
		for key in cant_move.keys():
			for a_id in cant_move[key]:
				next_positions_d[key][a_id] = self._agents[a_id].pos
	
	def aggregate_obs(self, hunter_obs: List, prey_obs: List) -> List:
		final_obs = []
		for idx in range(len(hunter_obs)):
			ordered_hunter_obs = hunter_obs[idx].copy()
			for elem in hunter_obs[:idx]:
				ordered_hunter_obs += elem.copy()
			for elem in hunter_obs[idx + 1:]:
				ordered_hunter_obs += elem.copy()
			final_obs += [ordered_hunter_obs + prey_obs]
		return final_obs

	####################
	### MAIN METHODS ###
	####################
	def spawn_hunters(self, init_pos: Dict[str, Tuple[int, int]] = None):
		
		if init_pos is None:
			hunter_initial_pos = self._initial_pos[AgentType.HUNTER - 1]
			initial_pos_id = list(hunter_initial_pos.keys())
			if len(hunter_initial_pos) > 0:
				for hunter in self._hunter_ids:
					if hunter in initial_pos_id:
						self._agents[hunter].pos = hunter_initial_pos[hunter]
						self._agents[hunter].alive = True
						
						self._field[hunter_initial_pos[hunter][0], hunter_initial_pos[hunter][1]] = AgentType.HUNTER
					else:
						agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
						while self._field[agent_pos[0], agent_pos[1]] != 0:
							agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
						self._agents[hunter].pos = agent_pos
						self._agents[hunter].alive = True
						
						self._initial_pos[AgentType.HUNTER - 1][hunter] = agent_pos
						self._field[agent_pos[0], agent_pos[1]] = AgentType.HUNTER
			else:
				for hunter in self._hunter_ids:
					agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
					while self._field[agent_pos[0], agent_pos[1]] != 0:
						agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
					self._agents[hunter].pos = agent_pos
					self._agents[hunter].alive = True
					
					self._initial_pos[AgentType.HUNTER - 1][hunter] = agent_pos
					self._field[agent_pos[0], agent_pos[1]] = AgentType.HUNTER
		
		else:
			for hunter in self._hunter_ids:
				self._agents[hunter].pos = init_pos[hunter]
				self._agents[hunter].alive = True
				
				self._initial_pos[AgentType.HUNTER - 1][hunter] = init_pos[hunter]
				self._field[init_pos[hunter][0], init_pos[hunter][1]] = AgentType.HUNTER
		
	def spawn_preys(self, init_pos: Dict[str, Tuple[int, int]] = None):
		
		if init_pos is None:
			prey_initial_pos = self._initial_pos[AgentType.PREY - 1]
			initial_pos_id = list(prey_initial_pos.keys())
			if len(prey_initial_pos) > 0:
				for prey in self._prey_ids:
					if prey in initial_pos_id:
						self._agents[prey].pos = prey_initial_pos[prey]
						self._agents[prey].alive = True
						
						self._field[prey_initial_pos[prey][0], prey_initial_pos[prey][1]] = AgentType.PREY
					else:
						agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
						while self._field[agent_pos[0], agent_pos[1]] != 0:
							agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
						self._agents[prey].pos = agent_pos
						self._agents[prey].alive = True
						
						self._initial_pos[AgentType.PREY - 1][prey] = agent_pos
						self._field[agent_pos[0], agent_pos[1]] = AgentType.PREY
			else:
				for prey in self._prey_ids:
					agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
					while self._field[agent_pos[0], agent_pos[1]] != 0:
						agent_pos = (self._np_random.choice(self._field_size[0]), self._np_random.choice(self._field_size[1]))
					self._agents[prey].pos = agent_pos
					self._agents[prey].alive = True
					
					self._initial_pos[AgentType.PREY - 1][prey] = agent_pos
					self._field[agent_pos[0], agent_pos[1]] = AgentType.PREY
		
		else:
			for prey in self._prey_ids:
				self._agents[prey].pos = init_pos[prey]
				self._agents[prey].alive = True
				
				self._initial_pos[AgentType.PREY - 1][prey] = init_pos[prey]
				self._field[init_pos[prey][0], init_pos[prey][1]] = AgentType.PREY
	
	def make_array_obs(self) -> np.ndarray:
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			agent_rank = self._agents[agent_id].rank
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				hunter_obs += [agent_pos + [AgentType.HUNTER]]
			else:
				prey_obs += agent_pos + [AgentType.PREY]
		
		return np.array(self.aggregate_obs(hunter_obs, prey_obs))
	
	def make_grid_obs(self) -> np.ndarray:
		layer_size = (self._field_size[0] + 2 * self._hunter_sight + 1, self._field_size[1] + 2 * self._hunter_sight + 1)
		hunter_layer = np.zeros(layer_size)
		prey_layer = np.zeros(layer_size)
		free_layer = np.ones(layer_size)
		free_layer[:self._hunter_sight, :] = 0
		free_layer[-self._hunter_sight:, :] = 0
		free_layer[:, self._hunter_sight] = 0
		free_layer[:, -self._hunter_sight:] = 0
		
		for agent in self._agents.values():
			if agent.alive:
				if agent.agent_type == AgentType.HUNTER:
					hunter_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				else:
					prey_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				free_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 0
		
		hunter_obs = np.stack([hunter_layer, prey_layer, free_layer])
		# prey_obs = np.stack([prey_layer, hunter_layer, free_layer])
		padding = 2 * self._hunter_sight + 1
		
		return np.array([hunter_obs[:, self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding] for hunter_id in self._hunter_ids])
	
	def make_dqn_obs(self) -> np.ndarray:
		
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			agent_rank = self._agents[agent_id].rank
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				type_one_hot = [1, 0]
				hunter_obs += [agent_pos + type_one_hot]
			else:
				type_one_hot = [0, 1]
				prey_obs += agent_pos + type_one_hot
		
		return np.array(self.aggregate_obs(hunter_obs, prey_obs))
	
	def make_obs(self) -> np.ndarray:
		
		if self._use_layer_obs:
			return self.make_grid_obs()
		elif self._use_encoding:
			return self.make_dqn_obs()
		else:
			return self.make_array_obs()
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		if seed is not None:
			self.seed(seed)
		
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
		info = self.get_info()
		
		return obs, info
	
	def step(self, actions: ActType) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:

		self._env_timestep += 1
		
		# Attempt moving each agent
		next_positions = {'hunter': defaultdict(tuple), 'prey': defaultdict(tuple)}
		prev_positions = []
		for agent_id, agent_action in zip(self._agents, actions):
			if self._agents[agent_id].alive:																			# Verify agent is still in play
				agent_pos = self._agents[agent_id].pos
				agent_type = 'hunter' if agent_id.find('hunter') != -1 else 'prey'
				prev_positions.append((agent_id, agent_pos))
				if agent_action == Action.UP:
					next_positions[agent_type][agent_id] = (max(agent_pos[0] - 1, 0), agent_pos[1])
				elif agent_action == Action.DOWN:
					next_positions[agent_type][agent_id] = (min(agent_pos[0] + 1, self._field_size[0] - 1), agent_pos[1])
				elif agent_action == Action.LEFT:
					next_positions[agent_type][agent_id] = (agent_pos[0], max(agent_pos[1] - 1, 0))
				elif agent_action == Action.RIGHT:
					next_positions[agent_type][agent_id] = (agent_pos[0], min(agent_pos[1] + 1, self._field_size[1] - 1))
				else:
					next_positions[agent_type][agent_id] = (agent_pos[0], agent_pos[1])
		
		# Resolve collisions
		self.resolve_collisions(next_positions, prev_positions)
		
		# Update hunter positions
		for agent_id, next_pos in next_positions['hunter'].items():
			self._agents[agent_id].pos = next_pos
				
		# Update field with hunters
		new_field = np.zeros(self._field_size)
		for hunter in self._hunter_ids:
			hunter_pos = self._agents[hunter].pos
			new_field[hunter_pos[0], hunter_pos[1]] = AgentType.HUNTER
		
		# Get rewards before updating captured preys
		rewards = self.get_rewards(new_field)
		
		# Check for captures
		captured_prey = []
		for prey_id in self._prey_ids:
			prey_adj = self.adj_pos(self._agents[prey_id].pos)
			is_surrounded = sum([new_field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj]) >= self._n_catch
			if is_surrounded:
				captured_prey += [prey_id]
				self._agents[prey_id].pos = (-1, -1)
				self._agents[prey_id].alive = False
		
		# Update prey positions
		for agent_id, next_pos in next_positions['prey'].items():
			if self._agents[agent_id].alive:
				self._agents[agent_id].pos = next_pos
		
		# Update field with preys
		for prey in self._prey_ids:
			if self._agents[prey].alive:
				prey_pos = self._agents[prey].pos
				new_field[prey_pos[0], prey_pos[1]] = AgentType.PREY
		self._field = new_field
		
		# Remove captured preys from play
		if len(captured_prey) > 0:
			for prey_id in captured_prey:
				self._prey_ids.remove(prey_id)
		self._n_preys = int(len(self._prey_ids))
		
		return self.make_obs(), rewards, self.game_finished(), self.timeout(), self.get_info()
	
	
class TeamPursuitEnv(PursuitEnv):
	
	_teams_comp: Dict[str, List[str]]
	_initial_teams: Dict[str, List[str]]
	_team_objective: Dict[str, str]
	_initial_objectives: Dict[str, str]
	
	def __init__(self, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]], field_size: Tuple[int, int], hunter_sight: int, teams: Dict[str, List[str]],
				 n_catch: int = 4, max_steps: int = 250, use_encoding: bool = False, dead_preys: List[bool] = None, target_preys: List[str] = None,
				 use_layer_obs: bool = False, agent_centered: bool = False):
		
		super().__init__(hunters, preys, field_size, hunter_sight, n_catch, max_steps, use_encoding, dead_preys, target_preys, use_layer_obs, agent_centered)
		
		self._teams_comp = dict()
		self._team_objective = dict()
		self._initial_teams = dict()
		self._initial_objectives = dict()
		for key in teams.keys():
			self._teams_comp[key] = teams[key].copy()
			self._initial_teams[key] = teams[key].copy()
			self._team_objective[key] = ''
			self._initial_objectives[key] = '-1'
			for a_id in teams[key]:
				self.agents[a_id].team = key
	
	@property
	def teams(self) -> Dict[str, List[str]]:
		return self._teams_comp
	
	@property
	def team_objectives(self) -> Dict[str, str]:
		return self._team_objective
	
	def _get_observation_space(self) -> Space:
		
		if self._use_layer_obs:
			# grid observation space
			grid_shape = (1 + 2 * self._hunter_sight, 1 + 2 * self._hunter_sight)
			
			# hunters layer: hunter positions
			hunters_min = np.zeros(grid_shape, dtype=np.int32)
			hunters_max = np.ones(grid_shape, dtype=np.int32)
			
			# preys layer: prey positions
			preys_min = np.zeros(grid_shape, dtype=np.int32)
			preys_max = np.ones(grid_shape, dtype=np.int32)
			
			# occupancy layer: i the cell available
			access_min = np.zeros(grid_shape, dtype=np.int32)
			access_max = np.ones(grid_shape, dtype=np.int32)
			
			# team layer: position of team members
			team_min = np.zeros(grid_shape, dtype=np.int32)
			team_max = np.ones(grid_shape, dtype=np.int32)
			
			# objective layer
			objective_min = np.zeros(grid_shape, dtype=np.int32)
			objective_max = np.ones(grid_shape, dtype=np.int32)
			
			# total layer
			min_obs = np.stack([hunters_min, preys_min, access_min, team_min, objective_min])
			max_obs = np.stack([hunters_max, preys_max, access_max, team_max, objective_max])
		
		else:
			n_teams = len(self._teams_comp.keys())
			if self._use_encoding:
				min_obs = [[-1, -1, 0, 0] * self._n_preys + ([-1, -1, 0, 0] + [0] * (n_teams + 1) + [0] * (self.n_preys + 1)) * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 1, 1] * self._n_preys +
						   ([self._field_size[0], self._field_size[1], 1, 1] + [1] * (n_teams + 1) + [1] * (self.n_preys + 1)) * self._n_hunters] * self._n_hunters
			else:
				min_obs = [[-1, -1, 2] * self._n_preys + [-1, -1, 1, 0, 0] * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 2] * self._n_preys +
						   [self._field_size[0], self._field_size[1], 1, n_teams, self.n_preys] * self._n_hunters] * self._n_hunters
			
		return Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)
	
	def aggregate_obs(self, hunter_obs: List, prey_obs: List) -> List:
		final_obs = []
		for team in self._teams_comp.keys():
			members_idx = [self._hunter_ids.index(m_id) for m_id in self._teams_comp[team]]
			ordered_hunter_obs = []
			for idx in members_idx:
				ordered_hunter_obs += hunter_obs[idx]
			final_obs += [ordered_hunter_obs + prey_obs]
		return final_obs
	
	def make_array_obs(self) -> np.ndarray:
		hunter_obs = []
		prey_obs = []
		
		for a_id in self._agents.keys():
			agent_pos = list(self._agents[a_id].pos)
			if self._agents[a_id].agent_type == AgentType.HUNTER:
				all_preys = [agent.agent_id for agent in self._agents.values() if agent.agent_type == AgentType.PREY]
				agent_team = list(self._teams_comp.keys()).index(self._agents[a_id].team)
				team_objective = all_preys.index(self._team_objective[self._agents[a_id].team])
				hunter_obs += [agent_pos + [AgentType.HUNTER] + [agent_team]+ [team_objective]]
			else:
				prey_obs += agent_pos + [AgentType.PREY]
		
		return np.array(self.aggregate_obs(hunter_obs, prey_obs))
	
	def make_dqn_obs(self) -> np.ndarray:
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				all_preys = [agent.agent_id for agent in self._agents.values() if agent.agent_type == AgentType.PREY]
				agent_team = [0] * len(self._teams_comp.keys())
				team_objective = [0] * len(all_preys)
				agent_team[list(self._teams_comp.keys()).index(self._agents[agent_id].team)] = 1
				team_objective[all_preys.index(self._team_objective[self._agents[agent_id].team])] = 1
				type_one_hot = [1, 0]
				hunter_obs += [agent_pos + type_one_hot + agent_team + team_objective]
			else:
				type_one_hot = [0, 1]
				prey_obs += agent_pos + type_one_hot
		
		return np.array(self.aggregate_obs(hunter_obs, prey_obs))
	
	def make_grid_obs(self) -> np.ndarray:
		layer_size = (self._field_size[0] + 2 * self._hunter_sight + 1, self._field_size[1] + 2 * self._hunter_sight + 1)
		hunter_layer = np.zeros(layer_size)
		prey_layer = np.zeros(layer_size)
		free_layer = np.ones(layer_size)
		team_layers = dict([[key, np.zeros(layer_size)] for key in self._teams_comp.keys()])
		objective_layers = dict([[key, np.zeros(layer_size)] for key in self._teams_comp.keys()])
		
		free_layer[:self._hunter_sight, :] = 0
		free_layer[-self._hunter_sight:, :] = 0
		free_layer[:, self._hunter_sight] = 0
		free_layer[:, -self._hunter_sight:] = 0
		
		for agent in self._agents.values():
			if agent.alive:
				if agent.agent_type == AgentType.HUNTER:
					hunter_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
					team_layers[agent.team][agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				else:
					prey_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
					for key in self._team_objective.keys():
						if agent.agent_id == self._team_objective[key]:
							objective_layers[key][agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				free_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 0
		
		hunter_obs = np.stack([hunter_layer, prey_layer, free_layer])
		# prey_obs = np.stack([prey_layer, hunter_layer, free_layer])
		padding = 2 * self._hunter_sight + 1
		
		return np.array([np.vstack([hunter_obs[:, self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding],
						 [team_layers[self.agents[hunter_id].team][self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding]],
						 [objective_layers[self.agents[hunter_id].team][self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding]]]) for hunter_id in self._hunter_ids])
	
	def setup_objectives(self, objectives: Dict[str, str]) -> None:
		
		team_ids = list(self._teams_comp.keys())
		for key in objectives.keys():
			if key in team_ids:
				self._team_objective[key] = objectives[key]
				self._initial_objectives[key] = objectives[key]
			else:
				print(colored('Team id %s is not a valid id.' % key, 'yellow'))
			
	def update_teams(self, new_teams: Dict[str, List[str]], remove_teams: Dict[str, List[str]] = None) -> None:
	
		existing_teams = list(self._teams_comp.keys())
		for key in new_teams.keys():
			self._teams_comp[key] = new_teams[key].copy()
			for a_id in self._teams_comp[key]:
				self.agents[a_id].team = key
		
		if remove_teams is not None:
			for key in remove_teams.keys():
				if key in existing_teams:
					for agent in remove_teams[key]:
						self._teams_comp[key].remove(agent)
				else:
					print(colored('Team id %s is not a valid id.' % key, 'yellow'))
	
	def update_objectives(self, new_objectives: Dict[str, str]) -> None:
		for key in new_objectives.keys():
			self._team_objective[key] = new_objectives[key]
	
	def correct_prey(self, agent_id: str, prey_id: str):
	
		for team_id in self._teams_comp.keys():
			if agent_id in self._teams_comp[team_id]:
				return prey_id == self._team_objective[team_id]

		return False
	
	def get_rewards(self, field: np.ndarray) -> np.ndarray:
		
		# If game ends and there are preys to be caught
		if self._env_timestep > self._env_timestep and self._n_preys > 0:
			rewards = [-EVADE_REWARD] * self._n_hunters
			for agent in self._agents:
				if self._agents[agent].agent_type == AgentType.PREY:
					rewards += [EVADE_REWARD]
			return np.array(rewards)
		
		# When all preys are caught, hunters get max reward and preys get max penalty
		if self._n_preys == 0:
			rewards = [CATCH_ALL_REWARD] * self._n_hunters
			for agent in self._agents:
				if self._agents[agent].agent_type == AgentType.PREY:
					rewards += [-CATCH_ALL_REWARD]
			return np.array(rewards)
		
		# By default, hunters get a zero or negative reward for moving
		rewards_hunters = [MOVE_REWARD] * self._n_hunters
		rewards_preys = []
		all_hunter_ids = [agent_id for agent_id in self._agents.keys() if self._agents[agent_id].agent_type == AgentType.HUNTER]
		all_prey_ids = [agent_id for agent_id in self._agents.keys() if self._agents[agent_id].agent_type == AgentType.PREY]
		for prey in all_prey_ids:
			if prey in self._prey_ids:
				prey_pos = self._agents[prey].pos
				prey_adj = self.adj_pos(prey_pos)
				n_surround_hunters = sum([self.correct_prey(hunter_id, prey) and self.agents[hunter_id].pos in prey_adj for hunter_id in self._hunter_ids])
				is_surrounded = n_surround_hunters >= self._n_catch
				# If prey is surrounded, gets a caught penalty and all hunters boxing it get a catching reward
				if is_surrounded:
					if self._n_preys == 1:
						for agent_id in all_hunter_ids:
							rewards_hunters[all_hunter_ids.index(agent_id)] = CATCH_ALL_REWARD
						rewards_preys += [-CATCH_ALL_REWARD]
					else:
						for agent_id in all_hunter_ids:
							if self._agents[agent_id].pos in prey_adj and self.correct_prey(agent_id, prey):
								rewards_hunters[all_hunter_ids.index(agent_id)] = CATCH_REWARD
						rewards_preys += [-CATCH_REWARD]
				# If prey is not surrounded, it gets a reward for being free and agents around it get a small reward
				else:
					rewards_preys += [-MOVE_REWARD]
					for agent_id in all_hunter_ids:
						if self._agents[agent_id].pos in prey_adj and self.correct_prey(agent_id, prey):
							rewards_hunters[all_hunter_ids.index(agent_id)] = n_surround_hunters * (CATCH_REWARD / self._n_catch)
			# If prey already caught, doesn't get reward
			else:
				rewards_preys += [0]
		
		return np.array(rewards_hunters + rewards_preys)
		
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		_, info = super().reset(seed=seed, options=options)
		for key in self._initial_teams.keys():
			self._teams_comp[key] = self._initial_teams[key].copy()
			if self._team_objective[key] == -1:
				self._team_objective[key] = self._initial_objectives[key]
			for a_id in self._teams_comp[key]:
				self.agents[a_id].team = key
		
		obs = self.make_obs()
		return obs, info
