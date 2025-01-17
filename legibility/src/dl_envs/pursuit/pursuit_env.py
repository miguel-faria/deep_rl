#! /usr/bin/env python

import numpy as np
import gymnasium

from gymnasium import Env
from typing import Tuple, List, Dict, Any, TypeVar, Union, Optional
from enum import IntEnum, Enum
from collections import defaultdict, namedtuple
from itertools import product
from gymnasium.utils import seeding
from gymnasium.spaces import Space, Box, MultiBinary, MultiDiscrete

from dl_envs.pursuit.agents.target_agent import TargetAgent
from dl_envs.pursuit.agents.greedy_prey import GreedyPrey
from dl_envs.pursuit.agents.random_prey import RandomPrey
from dl_envs.pursuit.agents.agent import Agent, AgentType

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

# Reward constants
TOUCH_FACTOR = 0.001
CATCH_ALL_FACTOR = 5


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
	action_space: MultiDiscrete
	observation_space: Union[MultiBinary, gymnasium.spaces.Tuple]
	
	_n_hunters: int
	_n_preys: int
	_n_preys_alive: int
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
	_n_need_catch: int
	_freeze_pos: bool
	
	metadata = {'render.modes': ['human', 'rgb_array']}
	
	def __init__(self, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]], field_size: Tuple[int, int], hunter_sight: int, n_catch: int = 4,
				 max_steps: int = 250, use_encoding: bool = False, dead_preys: List[bool] = None, use_layer_obs: bool = False, agent_centered: bool = False,
				 catch_reward: float = 1.0, render_mode: List[str] = None, freeze_pos: bool = False):
		
		self._prey_ids = [prey[0] for prey in preys]
		self._hunter_ids = [hunter[0] for hunter in hunters]
		self._prey_alive_ids = []
		self._n_hunters = int(len(hunters))
		self._n_preys = int(len(preys))
		self._n_preys_alive = 0
		self._field_size = field_size
		self._field = np.zeros(field_size)
		self._hunter_sight = hunter_sight
		self._max_timesteps = max_steps
		self._n_need_catch = n_catch
		self._dead_preys = [dead_preys.copy() if dead_preys is not None else [False] * self._n_preys_alive]
		self._use_encoding = use_encoding
		self._center_agent = agent_centered
		self._use_layer_obs = use_layer_obs
		self._freeze_pos = freeze_pos
		
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
		self.action_space = MultiDiscrete([n_actions] * (self._n_hunters + self._n_preys))
		self.observation_space = self._get_observation_space()
		self.reward_space = self.Rewards(catch=catch_reward, touch=(catch_reward / max_steps), catch_all=(catch_reward * CATCH_ALL_FACTOR),
										 evade=(catch_reward * CATCH_ALL_FACTOR), caught=(-1 * catch_reward), move=0.0)
		self._render = None
		self._rendering_initialized = False
		if render_mode is None:
			self.metadata = {"render_modes": ['human']}
			self._show_viewer = True
			self.render_mode = 'human'
		else:
			self.metadata = {"render_modes": [render_mode]}
			self._show_viewer = 'human' in render_mode
			self.render_mode = 'rgb_array' if 'rgb_array' in render_mode else 'human'
		
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
	def n_preys_alive(self) -> int:
		return self._n_preys_alive

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
	def n_catch(self) -> int:
		return self._n_need_catch
	
	#######################
	### UTILITY METHODS ###
	#######################
	def _get_observation_space(self) -> Union[MultiBinary, gymnasium.spaces.Tuple]:
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
	
	def sample_action(self) -> np.ndarray:
		return self.action_space.sample()
	
	def seed(self, seed=None):
		self._np_random, seed = seeding.np_random(seed)
		self.action_space.seed(seed)
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
	
	def get_hunters_in_pos(self, positions: List[Tuple[int, int]]) -> List[str]:
		hunters_ids = []
		for h_id in self.hunter_ids:
			if self.agents[h_id].pos in positions:
				hunters_ids.append(h_id)
		
		return hunters_ids
	
	def get_agent_pos(self, pos: Tuple[int, int]) -> Agent:
		for agent_id in self._agents.keys():
			if self._agents[agent_id].pos == pos:
				return self._agents[agent_id]
	
	def get_rewards(self, field: np.ndarray, captured_preys: List[str]) -> np.ndarray:
		
		# If game ends and there are preys to be caught
		if self._env_timestep > self._max_timesteps and self._n_preys_alive > 0:
			rewards = [-self.reward_space.evade] * self._n_hunters + [self.reward_space.evade] * len(self._prey_ids)
			return np.array(rewards)
		
		n_captured_preys = len(captured_preys)
		if n_captured_preys > 0:
			if self._n_preys_alive == 0:						# When all preys are caught, hunters get max reward and preys get max penalty
				rewards = [self.reward_space.catch_all] * self._n_hunters + [-self.reward_space.catch_all] * self._n_preys
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
							rewards_hunters[self._hunter_ids.index(agent_id)] = n_surround_hunters * self.reward_space.touch
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
						rewards_hunters[self._hunter_ids.index(agent_id)] = n_surround_hunters * self.reward_space.touch
		
		return np.array(rewards_hunters + rewards_preys)
	
	def env_finished(self) -> bool:
		return self._n_preys_alive <= 0
	
	def timeout(self) -> bool:
		return self._env_timestep > self._max_timesteps
	
	def get_info(self) -> dict[str, Any]:
		return {'preys_left': self._n_preys_alive, 'timestep': self._env_timestep}
	
	def resolve_collisions(self, next_positions_d: Dict) -> None:
		next_positions = {}
		for at in next_positions_d.keys():
			for a_id, a_pos in next_positions_d[at].items():
				if a_pos in next_positions.keys():
					next_positions[a_pos].append(a_id)
				else:
					next_positions[a_pos] = [a_id]
		all_unique = False
		while not all_unique:
			all_unique = True
			for a_pos in list(next_positions.keys()):
				a_ids = next_positions[a_pos]
				if len(a_ids) > 1:
					all_unique = False
					for a_id in a_ids:
						next_positions_d['hunter' if self._agents[a_id].agent_type == AgentType.HUNTER else 'prey'][a_id] = self._agents[a_id].pos
						next_positions[a_pos].remove(a_id)
						if self._agents[a_id].pos in next_positions.keys():
							next_positions[self._agents[a_id].pos].append(a_id)
						else:
							next_positions[self._agents[a_id].pos] = [a_id]

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
	def get_full_env_log(self) -> str:
	
		log = 'Environment state:\nHunter\'s states:\n'
		for hunter_id in self.hunter_ids:
			hunter = self._agents[hunter_id]
			log += '\t- hunter %s at (%d, %d) is %s\n' % (hunter_id, hunter.pos[0], hunter.pos[1], "alive" if hunter.alive else "dead")
		
		log += 'Prey\'s states:\n'
		for prey_id in self.prey_ids:
			prey = self._agents[prey_id]
			log += '\t- prey %s at (%d, %d) is %s\n' % (prey_id, prey.pos[0], prey.pos[1], "alive" if prey.alive else "dead")
		
		log += 'Field state:\n%s\n' % str(self.field)
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._env_timestep, self.env_finished(), self.timeout())
		
		return log
	
	def get_env_log(self) -> str:
		
		log = 'Environment state:\nHunter\'s states:\n'
		for hunter_id in self.hunter_ids:
			hunter = self._agents[hunter_id]
			log += '\t- hunter %s at (%d, %d) is %s\n' % (hunter_id, hunter.pos[0], hunter.pos[1], "alive" if hunter.alive else "dead")
		
		log += 'Prey\'s states:\n'
		for prey_id in self.prey_ids:
			prey = self._agents[prey_id]
			log += '\t- prey %s at (%d, %d) is %s\n' % (prey_id, prey.pos[0], prey.pos[1], "alive" if prey.alive else "dead")
		
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._env_timestep, self.env_finished(), self.timeout())
		
		return log
	
	def spawn_hunters(self, init_pos: Dict[str, Tuple[int, int]] = None):
		
		if init_pos is None:
			valid_pos = list(product(range(self._field_size[0]), range(self._field_size[1])))
			for pos in np.transpose(np.nonzero(self._field)):
				valid_pos.remove(tuple(pos))
				
			for hunter in self._hunter_ids:
				hunter_pos = tuple(self._np_random.choice(valid_pos))
				while self._field[hunter_pos[0], hunter_pos[1]] != 0 or any([self._field[pos[0], pos[1]] != 0 for pos in self.adj_pos(hunter_pos)]):
					valid_pos.remove(hunter_pos)
					hunter_pos = tuple(self._np_random.choice(valid_pos))
				self._agents[hunter].pos = hunter_pos
				self._agents[hunter].alive = True
				self._field[hunter_pos[0], hunter_pos[1]] = AgentType.HUNTER
				valid_pos.remove(hunter_pos)
		
		else:
			for hunter in self._hunter_ids:
				self._agents[hunter].pos = init_pos[hunter]
				self._agents[hunter].alive = True
				
				self._field[init_pos[hunter][0], init_pos[hunter][1]] = AgentType.HUNTER
		
	def spawn_preys(self, init_pos: Dict[str, Tuple[int, int]] = None):
		
		if init_pos is None:
			self._prey_alive_ids = self._prey_ids.copy()
			valid_pos = list(product(range(self._field_size[0]), range(self._field_size[1])))
			for pos in np.transpose(np.nonzero(self._field)):
				valid_pos.remove(tuple(pos))
			
			for prey in self._prey_ids:
				prey_pos = tuple(self._np_random.choice(valid_pos))
				n_free_adj = len(set(self.adj_pos(prey_pos)))
				while self._field[prey_pos[0], prey_pos[1]] != 0 or n_free_adj < self._n_need_catch or any([self._field[pos[0], pos[1]] != 0 for pos in self.adj_pos(prey_pos)]):
					valid_pos.remove(prey_pos)
					prey_pos = tuple(self._np_random.choice(valid_pos))
				self._agents[prey].pos = prey_pos
				self._agents[prey].alive = True
				self._field[prey_pos[0], prey_pos[1]] = AgentType.PREY
				valid_pos.remove(prey_pos)

		else:
			for prey in self._prey_ids:
				self._agents[prey].pos = init_pos[prey]
				self._agents[prey].alive = True
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
		
		self._env_timestep = 0
		if self._freeze_pos:
			hunter_pos = dict([(hunter_id, self._agents[hunter_id].pos) for hunter_id in self._hunter_ids])
			prey_pos = dict([(prey_id, self._agents[prey_id].pos) for prey_id in self._prey_ids])
		
		else:
			hunter_pos = None
			prey_pos = None
		
		# reset field and agents positions
		self._field = np.zeros(self._field_size)
		self.spawn_hunters(hunter_pos)
		self.spawn_preys(prey_pos)
		
		obs = self.make_obs()
		info = self.get_info()
		
		return obs, info
	
	def step(self, actions: ActType) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:

		self._env_timestep += 1
		
		# Attempt moving each agent
		next_positions = {'hunter': defaultdict(tuple), 'prey': defaultdict(tuple)}
		for agent_id, agent_action in zip(self._agents, actions):
			if self._agents[agent_id].alive:																			# Verify agent is still in play
				agent_pos = self._agents[agent_id].pos
				agent_type = 'hunter' if agent_id.find('hunter') != -1 else 'prey'
				if agent_action == Action.STAY:
					next_positions[agent_type][agent_id] = agent_pos
				else:
					row_delta, col_delta = ActionDirection[Action(agent_action).name].value
					next_pos = min(max(agent_pos[0] + row_delta, 0), self._field_size[0] - 1), min(max(agent_pos[1] + col_delta, 0), self._field_size[1] - 1)
					next_positions[agent_type][agent_id] = next_pos
		
		# Resolve collisions
		self.resolve_collisions(next_positions)
		
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
			is_surrounded = sum([new_field[pos[0], pos[1]] == AgentType.HUNTER for pos in prey_adj]) >= self._n_need_catch
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
		
		# Update number of preys alive
		self._n_preys_alive = int(len(self._prey_alive_ids))
		
		# Get rewards
		rewards = self.get_rewards(new_field, captured_prey)
		
		return self.make_obs(), rewards, self.env_finished(), self.timeout(), self.get_info()
	
	def render(self) -> np.ndarray | list[np.ndarray] | None:
		if not self._rendering_initialized:
			try:
				from .render import Viewer
				rows, cols = self.field_size
				self._render = Viewer((rows, cols), visible=self._show_viewer)
				self._rendering_initialized = True
			except Exception as e:
				print('Caught exception %s when trying to import Viewer class.' % str(e.args))

		return self._render.render(self, return_rgb_array=(self.render_mode == 'rgb_array'))
	
	def close_render(self):
		if self._render is not None:
			self._render.close()
			self._render = None
	
	def close(self):
		super().close()
		self.close_render()
	
	
class TargetPursuitEnv(PursuitEnv):
	
	_target_id: str
	_target_caught: bool
	
	def __init__(self, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]], field_size: Tuple[int, int], hunter_sight: int, target_id: str, n_catch: int = 4,
				 max_steps: int = 250, use_encoding: bool = False, dead_preys: List[bool] = None, use_layer_obs: bool = False, agent_centered: bool = False,
				 catch_reward: float = 1.0, render_mode: List[str] = None, freeze_pos: bool = False):
		
		super().__init__(hunters, preys, field_size, hunter_sight, n_catch, max_steps, use_encoding, dead_preys, use_layer_obs, agent_centered,
						 catch_reward, render_mode, freeze_pos)
		self._target_caught = False
		self._target_id = target_id
	
	###########################
	### GETTERS AND SETTERS ###
	###########################
	@property
	def target(self) -> str:
		return self._target_id
	
	@target.setter
	def target(self, new_target: str) -> None:
		self._target_id = new_target
	
	#######################
	### UTILITY METHODS ###
	#######################
	def _get_observation_space(self) -> Union[MultiBinary, gymnasium.spaces.Tuple]:
		if self._use_layer_obs:
			# grid observation space
			grid_shape = (1 + 2 * self._hunter_sight, 1 + 2 * self._hunter_sight)
			
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
	
	def env_finished(self) -> bool:
		return self.target not in self._prey_alive_ids
	
	####################
	### MAIN METHODS ###
	####################
	def get_full_env_log(self) -> str:
	
		log = 'Environment state:\nPlayer\'s states:\n'
		for hunter_id in self.hunter_ids:
			hunter = self._agents[hunter_id]
			log += '\t- hunter %s at (%d, %d) is %s\n' % (hunter_id, hunter.pos[0], hunter.pos[1], "alive" if hunter.alive else "dead")
		
		log += 'Food\'s states:\n'
		for prey_id in self.prey_ids:
			prey = self._agents[prey_id]
			log += '\t- prey %s at (%d, %d) is %s\n' % (prey_id, prey.pos[0], prey.pos[1], "alive" if prey.alive else "dead")
		
		log += 'Hunter\'s current target: %s\n' % str(self._target_id)
		log += 'Field state:\n%s\n' % str(self.field)
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._env_timestep, self.env_finished(), self.timeout())
		
		return log
	
	def get_env_log(self) -> str:
		
		log = 'Environment state:\nPlayer\'s states:\n'
		for hunter_id in self.hunter_ids:
			hunter = self._agents[hunter_id]
			log += '\t- hunter %s at (%d, %d) is %s\n' % (hunter_id, hunter.pos[0], hunter.pos[1], "alive" if hunter.alive else "dead")
		
		log += 'Food\'s states:\n'
		for prey_id in self.prey_ids:
			prey = self._agents[prey_id]
			log += '\t- prey %s at (%d, %d) is %s\n' % (prey_id, prey.pos[0], prey.pos[1], "alive" if prey.alive else "dead")
		
		log += 'Hunter\'s current target: %s\n' % str(self._target_id)
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._env_timestep, self.env_finished(), self.timeout())
		
		return log
	
	def make_array_obs(self) -> np.ndarray:
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				hunter_obs += [agent_pos + [AgentType.HUNTER]]
			else:
				prey_obs += agent_pos + [AgentType.PREY]
		
		target_pos = self.agents[self._target_id].pos
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
		target_pos = self.agents[self._target_id].pos
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
		
		target_pos = self.agents[self._target_id].pos
		aggregate_obs = self.aggregate_obs(hunter_obs, prey_obs)
		
		return np.array([[aggregate_obs[idx] + target_pos] for idx in range(self.n_hunters)])
	
	def get_rewards(self, field: np.ndarray, captured_preys: List[str]) -> np.ndarray:
		
		if self._env_timestep > self._max_timesteps and self._n_preys_alive > 0:	# If timeout and preys are alive, preys get evade reward and hunters penalty
			rewards = [self.reward_space.move] * self._n_hunters + [self.reward_space.evade] * self.n_preys
			return np.array(rewards)
		
		n_captured_preys = len(captured_preys)
		if n_captured_preys > 0:
			rewards_hunters = [self.reward_space.move] * self._n_hunters			# By default, hunters get a reward for moving
			if self.agents[self._target_id].alive:
				prey_adj = self.adj_pos(self.agents[self._target_id].pos)
				for agent_id in self._hunter_ids:
					if self._agents[agent_id].pos in prey_adj:
						rewards_hunters[self._hunter_ids.index(agent_id)] = self.reward_space.touch
			else:
				if self._target_id in captured_preys:  								# If target prey was just captured, hunters boxing it get a catching reward
					self._target_caught = True
					prey_adj = self.adj_pos(self._agents[self._target_id].pos)
					for agent_id in self._hunter_ids:
						if self._agents[agent_id].pos in prey_adj:
							rewards_hunters[self._hunter_ids.index(agent_id)] = self.reward_space.catch
		else:
			rewards_hunters = [self.reward_space.move] * self._n_hunters
			prey_adj = self.adj_pos(self._agents[self._target_id].pos)
			for agent_id in self._hunter_ids:										# Check for hunters near the target prey and give them a small reward
				if self._agents[agent_id].pos in prey_adj:
					rewards_hunters[self._hunter_ids.index(agent_id)] = self.reward_space.touch
		
		return np.array(rewards_hunters)
	
	def spawn_preys(self, init_pos: Dict[str, Tuple[int, int]] = None):
		
		if init_pos is None:
			prey_ids = self.prey_ids.copy()
			self._prey_alive_ids = prey_ids
			valid_pos = list(product(range(self._field_size[0]), range(self._field_size[1])))
			for pos in np.transpose(np.nonzero(self._field)):
				valid_pos.remove(tuple(pos))
				
			for prey in self._prey_ids:
				prey_pos = tuple(self._np_random.choice(valid_pos))
				n_free_adj = len(set(self.adj_pos(prey_pos)))
				while self._field[prey_pos[0], prey_pos[1]] != 0 or n_free_adj < self._n_need_catch or any([self._field[pos[0], pos[1]] != 0 for pos in self.adj_pos(prey_pos)]):
					valid_pos.remove(prey_pos)
					prey_pos = tuple(self._np_random.choice(valid_pos))
				self._agents[prey].pos = prey_pos
				self._agents[prey].alive = True
				self._field[prey_pos[0], prey_pos[1]] = AgentType.PREY
				valid_pos.remove(prey_pos)

		else:
			assert self._target_id in init_pos.keys(), 'When giving starting positions for preys, target prey %s must be among them' % self._target_id
			for prey in init_pos.keys():
				self._agents[prey].pos = init_pos[prey]
				self._agents[prey].alive = True
				self._field[init_pos[prey][0], init_pos[prey][1]] = AgentType.PREY
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		obs, info = super().reset(seed=seed, options=options)
		self._target_caught = False
		if self._rendering_initialized:
			self._render.highlight = self.target
		return obs, info
	
	def step(self, actions: ActType) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
		
		self._env_timestep += 1
		self._target_caught = False
		
		# Attempt moving each agent
		next_positions = {'hunter': defaultdict(tuple), 'prey': defaultdict(tuple)}
		for agent_id, agent_action in zip(self._agents, actions):
			if self._agents[agent_id].alive:  # Verify agent is still in play
				agent_pos = self._agents[agent_id].pos
				agent_type = 'hunter' if self._agents[agent_id].agent_type == AgentType.HUNTER else 'prey'
				if agent_action == Action.STAY:
					next_positions[agent_type][agent_id] = agent_pos
				else:
					row_delta, col_delta = ActionDirection[Action(agent_action).name].value
					next_pos = min(max(agent_pos[0] + row_delta, 0), self._field_size[0] - 1), min(max(agent_pos[1] + col_delta, 0), self._field_size[1] - 1)
					next_positions[agent_type][agent_id] = next_pos
		
		# Resolve collisions
		self.resolve_collisions(next_positions)
		
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
		capturing_hunters = []
		preys_alive = self.prey_alive_ids.copy()
		if self._target_id in preys_alive:		# First check for target prey
			preys_alive.remove(self._target_id)
			prey_adj = self.adj_pos(self._agents[self._target_id].pos)
			hunters_capturing = [h_id for h_id in self.get_hunters_in_pos(prey_adj) if h_id not in capturing_hunters]
			if len(hunters_capturing) >= self._n_need_catch:
				captured_prey += [self._target_id]
				self._agents[self._target_id].alive = False
				self._prey_alive_ids.remove(self._target_id)
				for h_id in hunters_capturing:
					capturing_hunters.append(h_id)
		for prey_id in preys_alive:				# Second check for other preys
			prey_adj = self.adj_pos(self._agents[prey_id].pos)
			hunters_capturing = []
			for h_id in self.get_hunters_in_pos(prey_adj):
				if h_id not in capturing_hunters:
					hunters_capturing.append(h_id)
			is_surrounded = len(hunters_capturing) >= self._n_need_catch
			if is_surrounded:
				captured_prey += [prey_id]
				self._agents[prey_id].alive = False
				self._prey_alive_ids.remove(prey_id)
				for h_id in hunters_capturing:
					capturing_hunters.append(h_id)
		
		# Update prey positions
		for agent_id, next_pos in next_positions['prey'].items():
			if self._agents[agent_id].alive:
				self._agents[agent_id].pos = next_pos
		
		# Update field with preys
		for prey in self._prey_alive_ids:
			prey_pos = self._agents[prey].pos
			new_field[prey_pos[0], prey_pos[1]] = AgentType.PREY
		self._field = new_field
		
		# Update number of preys alive
		self._n_preys_alive = int(len(self._prey_alive_ids))
		
		# Get rewards
		rewards = self.get_rewards(new_field, captured_prey)
		
		info = self.get_info()
		finished = self.env_finished()
		obs = self.make_obs()
		
		if self._target_caught:
			info['caught_target'] = True
		else:
			info['caught_target'] = False
		
		return obs, rewards, finished, self.timeout(), info

	def render(self) -> np.ndarray | list[np.ndarray] | None:
		if not self._rendering_initialized:
			try:
				from .render import Viewer
				rows, cols = self.field_size
				self._render = Viewer((rows, cols), visible=self._show_viewer, highlight=self.target)
				self._rendering_initialized = True
			except Exception as e:
				print('Caught exception %s when trying to import Viewer class.' % str(e.args))

		return self._render.render(self, return_rgb_array=(self.render_mode == 'rgb_array'))
	
