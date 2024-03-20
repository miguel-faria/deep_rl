#! /usr/bin/env python

import numpy as np
import gymnasium

from gymnasium import Env
from typing import Tuple, List, Dict, Any, TypeVar
from enum import IntEnum, Enum
from collections import defaultdict, namedtuple
from gymnasium.utils import seeding
from gymnasium.spaces import MultiDiscrete, Space, Box, MultiBinary
from dl_envs.pursuit.agents.target_agent import TargetAgent
from dl_envs.pursuit.agents.greedy_prey import GreedyPrey
from dl_envs.pursuit.agents.random_prey import RandomPrey
from dl_envs.pursuit.agents.agent import Agent, AgentType

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

# Reward constants
# MOVE_REWARD = 0.0
# CATCH_REWARD = 5
# TOUCH_REWARD = 1
# CATCH_ALL_REWARD = CATCH_REWARD * 10
# EVADE_REWARD = 10
# CAUGHT_REWARD = -5


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
	
	Rewards = namedtuple('Reward', ['catch', 'touch', 'catch_all', 'evade', 'caught', 'move'])
	
	_n_hunters: int
	_n_preys: int
	_n_preys_alive : int
	_field_size: Tuple[int, int]
	_hunter_ids: List[str]
	_prey_ids: List[str]
	_prey_alive_ids: List[str]
	_agents: Dict[str, Agent]
	_field: np.ndarray
	_hunter_sight: int
	_initial_pos: List[Dict[str, Tuple[int, int]]]
	_env_timestep: int
	_max_timesteps: int
	_n_catch: int
	metadata = {'render.modes': ['human', 'rgb_array']}
	
	def __init__(self, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]], field_size: Tuple[int, int], hunter_sight: int, n_catch: int = 4,
				 max_steps: int = 250, use_encoding: bool = False, dead_preys: List[bool] = None, use_layer_obs: bool = False, agent_centered: bool = False,
				 catch_reward: float = 1.0):
		
		self._prey_ids = [prey[0] for prey in preys]
		self._hunter_ids = [hunter[0] for hunter in hunters]
		self._prey_alive_ids = []
		self._n_hunters = int(len(hunters))
		self._n_preys = int(len(preys))
		self._n_preys_alive = 0
		self._field_size = field_size
		self._field = np.zeros(field_size)
		self._hunter_sight = hunter_sight
		self._initial_pos = [{} for _ in range(len(AgentType))]
		self._max_timesteps = max_steps
		self._n_catch = n_catch
		self._dead_preys = [dead_preys.copy() if dead_preys is not None else [False] * self._n_preys_alive]
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
			if p_type == 1:
				self._agents[p_id] = GreedyPrey(p_id, AgentType.PREY, rank)
			elif p_type == 2:
				self._agents[p_id] = RandomPrey(p_id, AgentType.PREY, rank)
			else:
				self._agents[p_id] = Agent(p_id, AgentType.PREY, rank)
		
		n_actions = len(Action)
		self.action_space = gymnasium.spaces.Tuple([MultiDiscrete([n_actions] * self._n_hunters), MultiDiscrete([n_actions] * self._n_preys)])
		self.observation_space = self._get_observation_space()
		self.reward_space = self.Rewards(catch=catch_reward, touch=(catch_reward / 5), catch_all=(catch_reward * 5), evade=(catch_reward * 5),
										 caught=(-1 * catch_reward), move=0.0)
		
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
	def prey_alive_ids(self) -> List[str]:
		return self._prey_alive_ids
	
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
			
			return MultiBinary([self._n_hunters, 3, *grid_shape])
		
		else:
			if self._use_encoding:
				min_obs = [[-1, -1, 0, 0] * self._n_preys_alive + [-1, -1, 0, 0] * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 1, 1] * self._n_preys_alive +
						   [self._field_size[0], self._field_size[1], 1, 1] * self._n_hunters] * self._n_hunters
			else:
				min_obs = [[-1, -1, 2] * self._n_preys_alive + [-1, -1, 1] * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 2] * self._n_preys_alive +
						   [self._field_size[0], self._field_size[1], 1] * self._n_hunters] * self._n_hunters
		
			return gymnasium.spaces.Tuple([Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)] * self._n_hunters)
	
	def sample_action(self) -> int:
		return self.action_space.sample()
	
	def seed(self, seed=None):
		self._np_random, seed = seeding.np_random(seed)
		self.action_space.seed(seed)
		if isinstance(self.action_space, gymnasium.spaces.Tuple):
			for idx in range(len(self.action_space)):
				self.action_space[idx].seed(seed)
		self.observation_space.seed(seed)
		if isinstance(self.observation_space, gymnasium.spaces.Tuple):
			for idx in range(len(self.observation_space)):
				self.observation_space[idx].seed(seed)
	
	def reset_init_pos(self) -> None:
		self._initial_pos = [{} for _ in range(len(AgentType))]
	
	def reset_timestep(self) -> None:
		self._env_timestep = 0
	
	def adj_pos(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
		return list({(max(pos[0] - 1, 0), pos[1]), (min(pos[0] + 1, self._field_size[0] - 1), pos[1]),
					 (pos[0], max(pos[1] - 1, 0)), (pos[0], min(pos[1] + 1, self._field_size[1] - 1))})
	
	def get_rewards(self, field: np.ndarray, captured_preys: List[str]) -> np.ndarray:
		
		# If game ends and there are preys to be caught
		if self._env_timestep > self._max_timesteps and self._n_preys_alive > 0:
			rewards = [self.reward_space.move] * self._n_hunters + [self.reward_space.evade] * len(self._prey_ids)
			return np.array(rewards)
		
		n_captured_preys = len(captured_preys)
		if n_captured_preys > 0:
			if self._n_preys_alive == 1:						# When all preys are caught, hunters get max reward and preys get max penalty
				rewards = [self.reward_space.catch_all] * self._n_hunters
				for agent in self._agents:
					if self._agents[agent].agent_type == AgentType.PREY:
						rewards += [-self.reward_space.catch_all]
				return np.array(rewards)
			
			rewards_hunters = [self.reward_space.move] * self._n_hunters	# By default, hunters get a reward for moving
			rewards_preys = []
			for prey_id in self._prey_ids:
				if prey_id in self._prey_alive_ids:						# If prey is alive it gets a reward for being free and agents around it get a small reward
					prey_pos = self._agents[prey_id].pos
					prey_adj = self.adj_pos(prey_pos)
					rewards_preys += [-self.reward_space.move]
					n_surround_hunters = sum([field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj])
					for agent_id in self._hunter_ids:
						if self._agents[agent_id].pos in prey_adj:
							rewards_hunters[self._hunter_ids.index(agent_id)] = n_surround_hunters * (self.reward_space.catch / self._n_catch)
				else:
					rewards_preys += [-self.reward_space.catch]			# If prey is dead gets a caught penalty
					if prey_id in captured_preys:						# If prey was just captured, all hunters boxing it get a catching reward
						prey_pos = self._agents[prey_id].pos
						prey_adj = self.adj_pos(prey_pos)
						for agent_id in self._hunter_ids:
							if self._agents[agent_id].pos in prey_adj:
								rewards_hunters[self._hunter_ids.index(agent_id)] = self.reward_space.catch
		else:
			rewards_preys = [-self.reward_space.move] * self._n_preys_alive
			rewards_hunters = [self.reward_space.move] * self._n_hunters
			for prey_id in self._prey_alive_ids:				# Check if there are hunters near a prey
				prey_pos = self._agents[prey_id].pos
				prey_adj = self.adj_pos(prey_pos)
				n_surround_hunters = sum([field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj])
				for agent_id in self._hunter_ids:
					if self._agents[agent_id].pos in prey_adj:
						rewards_hunters[self._hunter_ids.index(agent_id)] = n_surround_hunters * (self.reward_space.catch / self._n_catch)
		
		return np.array(rewards_hunters + rewards_preys)
	
	def game_finished(self) -> bool:
		return self._n_preys_alive <= 0
	
	def timeout(self) -> bool:
		return self._env_timestep > self._max_timesteps
	
	def get_info(self) -> dict[str, Any]:
		return {'preys_left': self._n_preys_alive, 'timestep': self._env_timestep}
	
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
	def get_env_log(self) -> str:
	
		log = 'Environment state:\nPlayer\'s states:\n'
		for hunter_id in self.hunter_ids:
			hunter = self._agents[hunter_id]
			log += '\t- hunter %s at (%d, %d) is %s\n' % (hunter_id, hunter.pos[0], hunter.pos[1], "alive" if hunter.alive else "dead")
		
		log += 'Food\'s states:\n'
		for prey_id in self.prey_ids:
			prey = self._agents[prey_id]
			log += '\t- prey %s at (%d, %d) is %s\n' % (prey_id, prey.pos[0], prey.pos[1], "alive" if prey.alive else "dead")
		
		log += 'Field state:\n%s\n' % str(self.field)
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._env_timestep, self.game_finished(), self.timeout())
		
		return log
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
		if (self._n_hunters + self._n_preys_alive) < n_agents:
			for agent_id in self._agents.keys():
				self._agents[agent_id].alive = True
			self._prey_alive_ids = self._prey_ids.copy()
			self._n_preys_alive = self._n_preys
		
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
		
		# Check for captures
		captured_prey = []
		for prey_id in self._prey_alive_ids:
			prey_adj = self.adj_pos(self._agents[prey_id].pos)
			is_surrounded = sum([new_field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj]) >= self._n_catch
			if is_surrounded:
				captured_prey += [prey_id]
				self._agents[prey_id].alive = False
				self._prey_alive_ids.remove(prey_id)
		
		# Update prey positions
		for agent_id, next_pos in next_positions['prey'].items():
			if self._agents[agent_id].alive:
				self._agents[agent_id].pos = next_pos
		
		# Update field with preys
		for prey in self._prey_alive_ids:
			prey_pos = self._agents[prey].pos
			new_field[prey_pos[0], prey_pos[1]] = AgentType.PREY
		self._field = new_field
		
		# Get rewards before updating preys alive
		rewards = self.get_rewards(new_field, captured_prey)
		
		# Remove captured preys from play
		self._n_preys_alive = int(len(self._prey_alive_ids))
		
		return self.make_obs(), rewards, self.game_finished(), self.timeout(), self.get_info()
	
	
class TargetPursuitEnv(PursuitEnv):
	
	_target_list: List[str]
	_target_caught: bool
	_target_idx: int
	
	def __init__(self, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]], field_size: Tuple[int, int], hunter_sight: int, target_list: List[str],
				 n_catch: int = 4, max_steps: int = 250, use_encoding: bool = False, dead_preys: List[bool] = None, use_layer_obs: bool = False,
				 agent_centered: bool = False):
		
		super().__init__(hunters, preys, field_size, hunter_sight, n_catch, max_steps, use_encoding, dead_preys, use_layer_obs, agent_centered)
		self._target_idx = 0
		self._target_caught = False
		self._target_list = target_list.copy()
	
	###########################
	### GETTERS AND SETTERS ###
	###########################
	@property
	def target_list(self) -> List[str]:
		return self._target_list
	
	@property
	def target_idx(self) -> int:
		return self._target_idx
	
	@target_list.setter
	def target_list(self, new_lst: List[str]) -> None:
		self._target_list = new_lst.copy()
		self._target_idx = 0
		self._target_caught = False

	####################
	### MAIN METHODS ###
	####################
	def get_env_log(self) -> str:
	
		log = 'Environment state:\nPlayer\'s states:\n'
		for hunter_id in self.hunter_ids:
			hunter = self._agents[hunter_id]
			log += '\t- hunter %s at (%d, %d) is %s\n' % (hunter_id, hunter.pos[0], hunter.pos[1], "alive" if hunter.alive else "dead")
		
		log += 'Food\'s states:\n'
		for prey_id in self.prey_ids:
			prey = self._agents[prey_id]
			log += '\t- prey %s at (%d, %d) is %s\n' % (prey_id, prey.pos[0], prey.pos[1], "alive" if prey.alive else "dead")
		
		log += 'Hunter\'s current target: %s\n' % str(self._target_list[self._target_idx])
		log += 'Field state:\n%s\n' % str(self.field)
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._env_timestep, self.game_finished(), self.timeout())
		
		return log
	
	def _get_observation_space(self) -> Space:
		if self._use_layer_obs:
			# grid observation space
			grid_shape = (1 + 2 * self._hunter_sight, 1 + 2 * self._hunter_sight)
			
			# hunters layer: hunters positions
			# hunters_min = np.zeros(grid_shape, dtype=np.int32)
			# hunters_max = np.ones(grid_shape, dtype=np.int32)
			
			# preys layer: preys positions
			# preys_min = np.zeros(grid_shape, dtype=np.int32)
			# preys_max = np.ones(grid_shape, dtype=np.int32)
			
			# access layer: i the cell available
			# occupancy_min = np.zeros(grid_shape, dtype=np.int32)
			# occupancy_max = np.ones(grid_shape, dtype=np.int32)
			
			# target layer: marks the target position
			# target_min = np.zeros(grid_shape, dtype=np.int32)
			# target_max = np.ones(grid_shape, dtype=np.int32)
			
			# total layer
			# min_obs = np.stack([hunters_min, preys_min, occupancy_min, target_min])
			# max_obs = np.stack([hunters_max, preys_max, occupancy_max, target_max])
			
			# return MultiDiscrete(np.array([max_obs - min_obs + 1] * self._n_hunters), dtype=np.int32)
			return MultiBinary([self._n_hunters, 4, *grid_shape])
		
		else:
			if self._use_encoding:
				min_obs = [[-1, -1, 0, 0] * self._n_preys_alive + [-1, -1, 0, 0] * self._n_hunters] * self._n_hunters + [0, 0]
				max_obs = [[self._field_size[0], self._field_size[1], 1, 1] * self._n_preys_alive +
						   [self._field_size[0], self._field_size[1], 1, 1] * self._n_hunters] * self._n_hunters + [self._field_size[0], self._field_size[1]]
			else:
				min_obs = [[-1, -1, 2] * self._n_preys_alive + [-1, -1, 1] * self._n_hunters] * self._n_hunters + [0, 0]
				max_obs = [[self._field_size[0], self._field_size[1], 2] * self._n_preys_alive +
						   [self._field_size[0], self._field_size[1], 1] * self._n_hunters] * self._n_hunters + [self._field_size[0], self._field_size[1]]
		
			return gymnasium.spaces.Tuple([Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)] * self._n_hunters)
	
	def make_array_obs(self) -> np.ndarray:
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				hunter_obs += [agent_pos + [AgentType.HUNTER]]
			else:
				prey_obs += agent_pos + [AgentType.PREY]
		
		target_pos = [*self.agents[self._target_list[min(self._target_idx, len(self._target_list) - 1)]].pos]
		aggregate_obs = self.aggregate_obs(hunter_obs, prey_obs)
		
		return np.array([[aggregate_obs[idx] + target_pos] for idx in range(self.n_hunters)])
	
	def make_grid_obs(self) -> np.ndarray:
		layer_size = (self._field_size[0] + 2 * self._hunter_sight + 1, self._field_size[1] + 2 * self._hunter_sight + 1)
		hunter_layer = np.zeros(layer_size)
		prey_layer = np.zeros(layer_size)
		
		free_layer = np.ones(layer_size)
		free_layer[:self._hunter_sight, :] = 0
		free_layer[-self._hunter_sight:, :] = 0
		free_layer[:, self._hunter_sight] = 0
		free_layer[:, -self._hunter_sight:] = 0
		
		target_layer = np.zeros(layer_size)
		if self._target_idx < len(self._target_list) and self.agents[self._target_list[self._target_idx]].alive:
			target_pos = [*self.agents[self._target_list[self._target_idx]].pos]
			target_layer[target_pos[0] + self._hunter_sight, target_pos[1] + self._hunter_sight] = 1
		
		for agent in self._agents.values():
			if agent.alive:
				if agent.agent_type == AgentType.HUNTER:
					hunter_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				else:
					prey_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				free_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 0
		
		hunter_obs = np.stack([hunter_layer, prey_layer, free_layer, target_layer])
		# prey_obs = np.stack([prey_layer, hunter_layer, free_layer])
		padding = 2 * self._hunter_sight + 1
		
		return np.array([hunter_obs[:, self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding] for hunter_id in self._hunter_ids])
	
	def make_target_grid_obs(self, target_id: str) -> np.ndarray:
		layer_size = (self._field_size[0] + 2 * self._hunter_sight + 1, self._field_size[1] + 2 * self._hunter_sight + 1)
		hunter_layer = np.zeros(layer_size)
		prey_layer = np.zeros(layer_size)
		
		free_layer = np.ones(layer_size)
		free_layer[:self._hunter_sight, :] = 0
		free_layer[-self._hunter_sight:, :] = 0
		free_layer[:, self._hunter_sight] = 0
		free_layer[:, -self._hunter_sight:] = 0
		
		target_layer = np.zeros(layer_size)
		if self.agents[target_id].alive:
			target_pos = [*self.agents[target_id].pos]
			target_layer[target_pos[0] + self._hunter_sight, target_pos[1] + self._hunter_sight] = 1
		
		for agent in self._agents.values():
			if agent.alive:
				if agent.agent_type == AgentType.HUNTER:
					hunter_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				else:
					prey_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				free_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 0
		
		hunter_obs = np.stack([hunter_layer, prey_layer, free_layer, target_layer])
		# prey_obs = np.stack([prey_layer, hunter_layer, free_layer])
		padding = 2 * self._hunter_sight + 1
		
		return np.array([hunter_obs[:, self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding] for hunter_id in self._hunter_ids])
	
	def make_dqn_obs(self) -> np.ndarray:
		
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				type_one_hot = [1, 0]
				hunter_obs += [agent_pos + type_one_hot]
			else:
				type_one_hot = [0, 1]
				prey_obs += agent_pos + type_one_hot
		
		target_pos = [*self.agents[self._target_list[self._target_idx]].pos]
		aggregate_obs = self.aggregate_obs(hunter_obs, prey_obs)
		
		return np.array([[aggregate_obs[idx] + target_pos] for idx in range(self.n_hunters)])
	
	def get_rewards(self, field: np.ndarray, captured_preys: List[str]) -> np.ndarray:
		
		if self._env_timestep > self._max_timesteps and self._n_preys_alive > 0:	# If timeout and preys are alive, preys get evade reward and hunters penalty
			rewards = [self.reward_space.move] * self._n_hunters + [self.reward_space.evade] * len(self._prey_ids)
			return np.array(rewards)
		
		n_captured_preys = len(captured_preys)
		if n_captured_preys > 0:
			if self._n_preys_alive == 1:											# When all preys are caught, hunters get max reward and preys max penalty
				self._target_caught = True
				rewards = [self.reward_space.catch_all] * self._n_hunters + [-self.reward_space.catch_all] * len(self._prey_ids)
				return np.array(rewards)
		
			rewards_hunters = [self.reward_space.move] * self._n_hunters						# By default, hunters get a reward for moving
			rewards_preys = []
			target_prey_id = self._target_list[self._target_idx]
			for prey_id in self._prey_ids:
				if self._agents[prey_id].alive:
					rewards_preys += [-self.reward_space.move]									# If prey is alive, it gets a reward
					# if prey_id == target_prey_id:									# Hunters around the target prey get a small reward
					# 	prey_pos = self._agents[prey_id].pos
					# 	prey_adj = self.adj_pos(prey_pos)
					# 	n_surround_hunters = sum([field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj])
						# for agent_id in self._hunter_ids:
						# 	if self._agents[agent_id].pos in prey_adj:
						# 		rewards_hunters[self._hunter_ids.index(agent_id)] = n_surround_hunters * (TOUCH_REWARD / self._n_catch)
								# rewards_hunters[self._hunter_ids.index(agent_id)] = self.reward_space.touch
				else:
					rewards_preys += [-self.reward_space.catch]  								# If prey is dead, gets a caught penalty
					if prey_id in captured_preys and prey_id == target_prey_id:  	# If target prey was just captured, hunters boxing it get a catching reward
						self._target_caught = True
						prey_pos = self._agents[prey_id].pos
						prey_adj = self.adj_pos(prey_pos)
						for agent_id in self._hunter_ids:
							if self._agents[agent_id].pos in prey_adj:
								rewards_hunters[self._hunter_ids.index(agent_id)] = self.reward_space.catch
		else:
			rewards_preys = [-self.reward_space.move] * len(self.prey_ids)
			rewards_hunters = [self.reward_space.move] * self._n_hunters
			prey_id = self._target_list[self._target_idx]
			prey_pos = self._agents[prey_id].pos
			prey_adj = self.adj_pos(prey_pos)
			# n_surround_hunters = sum([field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj])
			# for agent_id in self._hunter_ids:										# Check for hunters near the target prey and give them a small reward
			# 	if self._agents[agent_id].pos in prey_adj:
					# rewards_hunters[self._hunter_ids.index(agent_id)] = n_surround_hunters * (TOUCH_REWARD / self._n_catch)
					# rewards_hunters[self._hunter_ids.index(agent_id)] = self.reward_space.touch
		
		return np.array(rewards_hunters + rewards_preys)
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		obs, info = super().reset(seed=seed, options=options)
		self._target_idx = 0
		self._target_caught = False
		
		return obs, info
	
	def step(self, actions: ActType) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
		
		self._env_timestep += 1
		
		# Attempt moving each agent
		next_positions = {'hunter': defaultdict(tuple), 'prey': defaultdict(tuple)}
		prev_positions = []
		for agent_id, agent_action in zip(self._agents, actions):
			if self._agents[agent_id].alive:  # Verify agent is still in play
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
		
		# Check for captures
		captured_prey = []
		for prey_id in self._prey_alive_ids:
			prey_adj = self.adj_pos(self._agents[prey_id].pos)
			is_surrounded = sum([new_field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj]) >= self._n_catch
			if is_surrounded:
				captured_prey += [prey_id]
				self._agents[prey_id].alive = False
				self._prey_alive_ids.remove(prey_id)
		
		# Update prey positions
		for agent_id, next_pos in next_positions['prey'].items():
			if self._agents[agent_id].alive:
				self._agents[agent_id].pos = next_pos
		
		# Update field with preys
		for prey in self._prey_alive_ids:
			prey_pos = self._agents[prey].pos
			new_field[prey_pos[0], prey_pos[1]] = AgentType.PREY
		self._field = new_field
		
		# Get rewards before updating preys alive
		rewards = self.get_rewards(new_field, captured_prey)
		
		# Remove captured preys from play
		self._n_preys_alive = int(len(self._prey_alive_ids))
		
		info = self.get_info()
		finished = self.game_finished()
		obs = self.make_obs()
		
		if self._target_caught:
			self._target_idx += 1
			info['caught_target'] = True
			info['real_obs'] = self.make_obs()
			self._target_caught = False
			if self._target_idx >= len(self._target_list):
				finished = True
		else:
			info['caught_target'] = False
			info['real_obs'] = None
		
		return obs, rewards, finished, self.timeout(), info

