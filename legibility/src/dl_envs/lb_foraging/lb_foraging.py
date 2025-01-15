import logging
import numpy as np
import gymnasium
import math

from collections import namedtuple, defaultdict
from enum import Enum, IntEnum
from itertools import product
from gymnasium import Env
from gymnasium.envs import Any
from gymnasium.utils import seeding
from gymnasium.spaces import MultiBinary, MultiDiscrete, Space, Box
from typing import Tuple, List, Dict, Optional, Union


class Action(IntEnum):
	NONE = 0
	NORTH = 1
	SOUTH = 2
	WEST = 3
	EAST = 4
	LOAD = 5


class Priority(IntEnum):
	NONE = 0
	RANDOM = 1
	ORDER = 2
	LEVEL = 3


class Direction(Enum):
	NONE = (0, 0)
	NORTH = (-1, 0)
	SOUTH = (1, 0)
	WEST = (0, -1)
	EAST = (0, 1)
	LOAD = (0, 0)


class CellEntity(IntEnum):
	# entity encodings for grid observations
	OUT_OF_BOUNDS = -1
	EMPTY = 0
	FOOD = 1
	AGENT = 2


class Food:
	def __init__(self):
		self._position = None
		self._level = None
		self._id = None
		self._picked = False

	def setup(self, position: Tuple[int, int], level: int, f_id: int):
		self._position = position
		self._level = level
		self._id = f_id

	@property
	def position(self) -> Tuple[int, int]:
		return self._position

	@property
	def level(self) -> int:
		return self._level

	@property
	def food_id(self) -> int:
		return self._id

	@property
	def picked(self) -> bool:
		return self._picked

	@position.setter
	def position(self, new_pos: Tuple[int, int]) -> None:
		self._position = new_pos

	@picked.setter
	def picked(self, new_val: bool) -> None:
		self._picked = new_val

	def deepcopy(self):
		new_food = Food()
		new_food.setup(self.position, self.level, self.food_id)
		new_food.picked = self.picked
		return new_food

	def __eq__(self, other):
		return (
				isinstance(other, Food)
				and self.position == other.position
				and self.level == other.level
				and self.food_id == other.food_id
				and self.picked == other.picked
		)

	def __hash__(self):
		return hash((self.level, self.food_id))

	def __str__(self):
		return "Food {} with level {} at {} is {}.".format(self.food_id, self.level, self.position, 'picked' if self.picked else 'not picked')

	def to_dict(self):
		return {
				"id": self.food_id,
				"position": self.position,
				"level": self.level,
				"picked": self.picked,
		}


class Player:
	def __init__(self, level: int=0):
		self._controller = None
		self._position = None
		self._level = level
		self._field_size = None
		self._score = None
		self._reward = 0
		self._history = None
		self._current_step = None
		self._id = None

	def setup(self, position: Tuple[int, int], level: int, field_size: Tuple[int, int], p_id: Union[int, str, None]):
		self._history = []
		self._position = position
		self._level = level
		self._field_size = field_size
		self._score = 0
		self._id = p_id

	def set_controller(self, controller):
		self._controller = controller

	def step(self, obs):
		return self.controller._step(obs)

	@property
	def player_id(self) -> int:
		return self._id

	@property
	def name(self) -> str:
		if self._controller:
			return self._controller.name
		else:
			return "Player"

	@property
	def position(self) -> Tuple[int, int]:
		return self._position

	@property
	def level(self) -> int:
		return self._level

	@property
	def controller(self):
		return self._controller

	@property
	def field_size(self) -> Tuple[int, int]:
		return self._field_size

	@property
	def history(self) -> List:
		return self._history

	@property
	def score(self) -> float:
		return self._score

	@property
	def reward(self) -> float:
		return self._reward

	@position.setter
	def position(self, new_pos: Tuple[int]):
		self._position = new_pos

	@score.setter
	def score(self, new_score: float):
		self._score = new_score

	@reward.setter
	def reward(self, new_reward: float):
		self._reward = new_reward

	@history.setter
	def history(self, new_history: List):
		self._history = new_history.copy()

	def deepcopy(self):
		new_player = Player(self._level)
		new_player.setup(self._position, self._level, self._field_size, self._id)
		return new_player

	def __eq__(self, other):
		return (
				isinstance(other, Player) and self.position == other.position and self.level == other.level and self.name == other.name and
				self.reward == other.reward and self.history == other.history and self.score == other.score and self.player_id == other.player_id and
				self.controller == other.controller
		)

	def __hash__(self):
		return hash((self.level, self.name, self.player_id))

	def __str__(self):
		return "Agent {} with level {} at {} has score {}.".format(self.name, self.level, self.position, self.score)

	def to_dict(self):
		return {
				"name": self.name,
				"position": self.position,
				"level": self.level,
				"score": self.score,
				"reward": self.reward,
				"id": self.player_id,
		}


class LBForagingEnv(Env):
	action_set = [Action.NONE, Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
	Observation = namedtuple("Observation", ["field", "foods", "players", "game_over", "sight", "current_step"])
	PlayerObservation = namedtuple( "PlayerObservation", ["position", "level", "history", "reward", "is_self"])  # reward is available only if is_self

	_foods: Optional[List[Food]]
	_game_over: Optional[bool]
	_render_initialized: Optional[bool]
	action_space: MultiDiscrete
	observation_space: Union[MultiBinary, gymnasium.spaces.Tuple]

	def __init__(self, n_players: int, max_player_level: int, field_size: Tuple[int, int], max_food: int, sight: int, max_episode_steps: int, force_coop: bool,
	             normalize_reward: bool = True, grid_observation: bool = False, penalty: float = 0.0, render_mode: List[str] = None, max_food_lvl: int = 0,
	             priority_mode: int = Priority.NONE, use_encoding: bool = False, agent_center: bool = False, use_render: bool = False):
		self.logger = logging.getLogger(__name__)
		self._players = [Player() for _ in range(n_players)]
		self._field_size = field_size
		self._field = np.zeros(field_size, np.int32)

		self._penalty = penalty
		self._max_spawn_food = max_food
		self._max_food_lvl = max_food_lvl
		self._foods = None
		self._food_spawned = 0
		self._max_player_level = max_player_level
		self._sight = sight
		self._force_coop = force_coop
		self._game_over = False
		self._current_step = 0
		self._rendering_initialized = False
		self._max_episode_steps = max_episode_steps
		self._render = None
		self._use_render = use_render
		self._priority_mode = priority_mode

		self._use_encoding = use_encoding
		self._agent_center = agent_center
		self._normalize_reward = normalize_reward
		self._grid_observation = grid_observation

		self.action_space = MultiDiscrete([len(self.action_set)] * n_players)
		self.observation_space = self._get_observation_space()
		self.reward_range = (0, self._max_food_lvl)
		self.seed()

		self._n_agents = n_players
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
	def field_size(self) -> Tuple:
		return self._field_size

	@property
	def rows(self) -> int:
		return self.field_size[0]

	@property
	def cols(self) -> int:
		return self.field_size[1]

	@property
	def game_over(self) -> bool:
		return self._game_over

	@property
	def field(self) -> np.ndarray:
		return self._field

	@property
	def players(self) -> List[Player]:
		return self._players

	@property
	def max_foods(self) -> int:
		return self._max_spawn_food

	@property
	def max_food_level(self) -> int:
		return self._max_food_lvl

	@property
	def n_players(self) -> int:
		return self._n_agents

	@property
	def max_player_level(self) -> int:
		return self._max_player_level

	@property
	def foods(self) -> List[Food]:
		return self._foods

	@property
	def timestep(self) -> int:
		return self._current_step

	@property
	def use_render(self) -> bool:
		return self._use_render

	@property
	def force_coop(self) -> bool:
		return self._force_coop

	@field.setter
	def field(self, new_field: np.ndarray) -> None:
		self._field = new_field

	@use_render.setter
	def use_render(self, new_val: bool) -> None:
		self._use_render = new_val

	#######################
	### UTILITY METHODS ###
	#######################
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
		return [seed]

	def _get_observation_space(self) -> Union[MultiBinary, gymnasium.spaces.Tuple]:
		"""
		The Observation Space for each agent.
		- the board (board_size^2) with foods
		- player description (x, y, level)*player_count
		"""
		if not self._grid_observation:
			field_x = self._field.shape[1]
			field_y = self._field.shape[0]

			max_food = self._max_spawn_food
			max_food_level = self._max_player_level * self._n_agents

			if self._use_encoding:
				min_obs = [-1, -1, 0, *[0] * max_food_level] * max_food + [-1, -1, 0] * self._n_agents
				max_obs = ([field_x - 1, field_y - 1, 1, *[1] * max_food_level] * max_food +
				           [field_x - 1, field_y - 1, 1, *[1] * self._max_player_level]) * self._n_agents
			else:
				min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self._players)
				max_obs = [field_x - 1, field_y - 1, max_food_level] * max_food + [field_x - 1, field_y - 1, self._max_player_level] * self._n_agents

			return gymnasium.spaces.Tuple([Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)] * self._n_agents)
		else:
			# grid observation space
			grid_shape = (1 + 2 * self._sight, 1 + 2 * self._sight)

			return MultiBinary([self._n_agents, 3, *grid_shape])	# Three overlapped observation layers (agents, food, occupied spaces)

	@classmethod
	def from_obs(cls, obs):
		players = []
		for p in obs.players:
			player = Player()
			player.setup(p.position, p.level, obs.field.shape, p.player_id)
			player.score = p.score if p.score else 0
			players.append(player)

		env = cls(players, None, None, None, None)
		env.field = np.copy(obs.field)
		env.current_step = obs.current_step
		env.sight = obs.sight

		return env

	def count_foods(self) -> int:
		return sum([1 if not food.picked else 0 for food in self._foods])

	def reset_timesteps(self) -> None:
		self._current_step = 0

	def get_adj_pos(self, row: int, col: int) -> List:
		return [(max(row - 1, 0) , col), (min(row + 1, self.rows), col), (row, max(col - 1, 0)), (row, min(col + 1, self.cols))]

	def neighborhood(self, row: int, col: int, distance: int = 1, ignore_diag: bool = False) -> np.array:
		if not ignore_diag:
			return self._field[max(row - distance, 0) : min(row + distance + 1, self.rows), max(col - distance, 0) : min(col + distance + 1, self.cols)]

		return np.array([list(self._field[max(row - distance, 0) : row, col]) + list(self._field[row + 1 : min(row + distance + 1, self.rows), col])] +
		                [list(self._field[row, max(col - distance, 0) : col]) + list(self._field[row, col + 1 : min(col + distance + 1, self.cols)])])

	def adjacent_foods(self, row: int, col: int) -> List:
		return [food for food in self._foods if
		        (abs(food.position[0] - row) == 1 and food.position[1] == col) or (food.position[0] == row and abs(food.position[1] - col) == 1)]

	def adjacent_players(self, row: int, col: int) -> List:
		return [player for player in self._players if
		        (abs(player.position[0] - row) == 1 and player.position[1] == col) or (player.position[0] == row and abs(player.position[1] - col) == 1)]

	def check_valid_food(self, row: int, col: int, check_distance: int = 2) -> bool:
		for x in product(range(max(0, row - check_distance), min(self.rows, row + check_distance + 1)),
		                 range(max(0, col - check_distance), min(self.cols, col + check_distance + 1))):
			if abs(x[0] - row) + abs(x[1] - col) == 2 and self.field[x[0], x[1]] == CellEntity.FOOD:
				return False

		return True

	def _is_empty_location(self, row: int, col: int) -> bool:
		if self.field[row, col] != CellEntity.EMPTY:
			return False

		return True

	@staticmethod
	def transform_to_neighborhood(center: Tuple[int, int], sight: int, position: Tuple[int, int]) -> Tuple:
		return position[0] - center[0] + min(sight, center[0]), position[1] - center[1] + min(sight, center[1])

	def get_center_pos(self, anchor_pos: Tuple[int, int]) -> Tuple[int, int]:
		rows, cols = (min(2 * self._sight + 1, self.rows), min(2 * self._sight + 1, self.cols))
		half_rows = int(rows / 2)
		half_cols = int(cols / 2)
		even_pos = math.floor(rows / 2) - 1
		odd_pos = math.floor(rows / 2)
		if half_rows & 1 and half_cols & 1:
			return odd_pos, odd_pos
		else:
			if half_rows & 1:
				return odd_pos, even_pos if anchor_pos[1] <= (self.cols / 2) else odd_pos
			elif half_cols & 1:
				return even_pos if anchor_pos[1] <= (self.rows / 2) else odd_pos, odd_pos
			else:
				return even_pos if anchor_pos[0] <= (self.rows / 2) else odd_pos, even_pos if anchor_pos[1] <= (self.cols / 2) else odd_pos

	def get_centered_pos(self, center: Tuple[int, int], position: Tuple[int, int]) -> Tuple[int, int]:
		transformed_pos = (position[0] - center[0], position[1] - center[1])
		return transformed_pos if max([abs(transformed_pos[0]), abs(transformed_pos[1])]) <= self._sight else None

	def resolve_collisions(self, attempt_pos: dict) -> None:
		for key, vals in attempt_pos.items():
			if len(vals) > 1:
				continue
			else:
				new_row, new_col = key
				agent = vals[0]
				self.field[agent.position[0], agent.position[1]] = CellEntity.EMPTY
				self.field[new_row, new_col] = CellEntity.AGENT
				agent.position = key

	def resolve_loading(self, loading_agents: set) -> None:
		remain_foods = [food for food in self._foods if not food.picked]

		for food in remain_foods:
			f_row, f_col = food.position
			food_lvl = food.level
			food_adj_pos = self.get_adj_pos(f_row, f_col)
			adj_agents = []
			adj_agents_lvl = 0
			for agent in loading_agents:
				if agent.position in food_adj_pos:
					adj_agents.append(agent)
					adj_agents_lvl += agent.level

			if adj_agents_lvl >= food_lvl:
				for agent in adj_agents:
					if self._normalize_reward:
						agent.reward = (agent.level / adj_agents_lvl) * food_lvl
					else:
						agent.reward = agent.level * food_lvl
					loading_agents.remove(agent)
				self.field[f_row, f_col] = CellEntity.EMPTY
				food.picked = True

	####################
	### MAIN METHODS ###
	####################
	def get_env_log(self) -> str:
		log = 'Environment state:\nPlayer\'s states:\n'
		for player in self.players:
			log += '\t- player %s has level %d is at (%d, %d) has score %f\n' % (player.player_id, player.level, player.position[0],
			                                                                     player.position[1], player.score)

		log += 'Food\'s states:\n'
		for food in self.foods:
			log += '\t- food %s has level %d is at (%d, %d) has %s\n' % (food.food_id, food.level, food.position[0], food.position[1],
			                                                             'been picked' if food.picked else 'not been picked')

		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._current_step, self.game_over,
		                                                                                 self._current_step > self._max_episode_steps)

		return log

	def get_full_env_log(self) -> str:
		log = 'Environment state:\nPlayer\'s states:\n'
		for player in self.players:
			log += '\t- player %s has level %d is at (%d, %d) has score %f\n' % (player.player_id, player.level, player.position[0],
			                                                                     player.position[1], player.score)

		log += 'Food\'s states:\n'
		for food in self.foods:
			log += '\t- food %s has level %d is at (%d, %d) has %s\n' % (food.food_id, food.level, food.position[0], food.position[1],
			                                                             'been picked' if food.picked else 'not been picked')

		log += 'Field state:\n%s\n' % str(self.field)
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._current_step, self.game_over,
		                                                                                 self._current_step > self._max_episode_steps)

		return log

	def spawn_food(self, max_food: int, max_level: int) -> None:
		food_count = 0
		min_level = max_level if self._force_coop else 1
		foods_spawn = []

		valid_pos = list(product(range(self.rows), range(self.cols)))
		for pos in np.transpose(np.nonzero(self._field)):
			valid_pos.remove(tuple(pos))

		while food_count < max_food:
			row, col = self._np_random.choice(valid_pos)

			# check if pos causes incompatibilities
			is_legal = self.check_valid_food(row, col)
			if is_legal:
				food_lvl = min_level if min_level == max_level else self.np_random.integers(min_level, max_level, endpoint=True)
				new_food = Food()
				new_food.setup((row, col), food_lvl, food_count + 1)
				foods_spawn.append(new_food)
				self.field[row, col] = CellEntity.FOOD
				food_count += 1
			valid_pos.remove((row, col))

		self._food_spawned = food_count
		self._foods = foods_spawn

	def spawn_players(self, player_levels: List = None) -> None:
		valid_pos = list(product(range(self.rows), range(self.cols)))
		for pos in np.transpose(np.nonzero(self._field)):
			valid_pos.remove(tuple(pos))

		players_spawn = 0
		for player_idx in range(len(self._players)):
			player = self._players[player_idx]
			row, col = self.np_random.choice(valid_pos)
			player_lvl = player_levels[player_idx] if player_levels is not None else self.np_random.integers(1, self._max_player_level, endpoint=True)
			player.setup((row, col), player_lvl, self.field_size, players_spawn + 1)
			player.reward = 0
			self.field[row, col] = CellEntity.AGENT
			valid_pos.remove(tuple([row, col]))
			players_spawn += 1

	def make_obs(self, player) -> Observation:
		return self.Observation(
				players=[
						self.PlayerObservation(
								position=a.position,
								level=a.level,
								is_self=a == player,
								history=a.history,
								reward=a.reward if a == player else None,
						)
						for a in self._players
						if self.get_centered_pos(player.position, a.position) is not None
				],
				foods=[food for food in self._foods if not food.picked and self.get_centered_pos(player.position, food.position) is not None],
				field=np.copy(self.neighborhood(*player.position, self._sight)),
				game_over=self.game_over,
				sight=self._sight,
				current_step=self._current_step,
		)

	def make_gym_obs(self) -> tuple[np.ndarray, np.ndarray, bool, bool, Dict]:

		rewards = np.zeros(self._n_agents)
		done = self.game_over
		force_stop = self._current_step > self._max_episode_steps
		info = {}

		if self._grid_observation:
			obs = self.make_grid_observations()
		else:
			obs = self.make_obs_array()

		for idx in range(self._n_agents):
			rewards[idx] = self._players[idx].reward

		return obs, rewards, done, force_stop, info

	def make_grid_observations(self) -> np.ndarray:
		layers_size = (self._field_size[0] + 2 * self._sight, self._field_size[1] + 2 * self._sight)
		agent_layer = np.zeros(layers_size)
		food_layer = np.zeros(layers_size)
		occupancy_layer = np.ones(layers_size)
		occupancy_layer[:self._sight, :] = 0
		occupancy_layer[-self._sight:, :] = 0
		occupancy_layer[:, :self._sight] = 0
		occupancy_layer[:, -self._sight:] = 0
		for a in self._players:
			pos = a.position
			agent_layer[pos[0] + self._sight, pos[1] + self._sight] = a.level
			occupancy_layer[pos[0] + self._sight, pos[1] + self._sight] = 0

		for f in self._foods:
			if not f.picked:
				pos = f.position
				food_layer[pos[0] + self._sight, pos[1] + self._sight] = f.level if not self._force_coop else 1
				occupancy_layer[pos[0] + self._sight, pos[1] + self._sight] = 0

		obs = np.stack([agent_layer, food_layer, occupancy_layer])
		padding = 2 * self._sight + 1
		return np.array([obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding] for a in self._players])

	def make_obs_array(self) -> np.ndarray:
		if self._grid_observation:
			self._grid_observation = False
			obs = np.array([np.zeros(self._get_observation_space().shape, dtype=np.int32)] * self._n_agents)
			self._grid_observation = True
		else:
			obs = np.array([np.zeros(self.observation_space[0].shape, dtype=np.int32)] * self._n_agents)
		raw_obs = [self.make_obs(p) for p in self._players]

		for idx in range(self._n_agents):
			p_obs = raw_obs[idx]
			seen_agents = tuple([p for p in p_obs.players if p.is_self] + [p for p in p_obs.players if not p.is_self])
			seen_foods = tuple(p_obs.foods)
			n_seen_foods = len(seen_foods)
			n_seen_agents = len(seen_agents)

			for i in range(self._max_spawn_food):
				if i < n_seen_foods:
					if self._agent_center:
						f_row, f_col = self.get_centered_pos(self._players[idx].position, seen_foods[i].position)
					else:
						f_row, f_col = seen_foods[i].position
					obs[idx][3 * i] = f_row
					obs[idx][3 * i + 1] = f_col
					obs[idx][3 * i + 2] = seen_foods[i].level
				else:
					obs[idx][3 * i] = -1
					obs[idx][3 * i + 1] = -1
					obs[idx][3 * i + 2] = 0

			for i in range(self._n_agents):
				if i < n_seen_agents:
					if self._agent_center:
						p_row, p_col = self.get_centered_pos(self._players[idx].position, seen_agents[i].position)
					else:
						p_row, p_col = seen_agents[i].position
					obs[idx][3 * self._max_spawn_food + 3 * i] = p_row
					obs[idx][3 * self._max_spawn_food + 3 * i + 1] = p_col
					obs[idx][3 * self._max_spawn_food + 3 * i + 2] = seen_agents[i].level
				else:
					obs[idx][3 * self._max_spawn_food + 3 * i] = -1
					obs[idx][3 * self._max_spawn_food + 3 * i + 1] = -1
					obs[idx][3 * self._max_spawn_food + 3 * i + 2] = 0
		return obs

	def make_obs_dqn_array(self) -> tuple[np.ndarray, np.ndarray, bool, bool, Dict]:

		obs = []
		rewards = np.zeros(self._n_agents)
		done = self.game_over
		force_stop = self._current_step > self._max_episode_steps
		raw_obs = [self.make_obs(p) for p in self._players]
		info = {}

		for a_idx in range(self._n_agents):
			food_obs = ([-1, -1, 1] + [0] * self._max_food_lvl) * self._max_spawn_food
			agent_obs = ([-1, -1, 1] + [0] * self._max_player_level) * self._n_agents
			p_obs = raw_obs[a_idx]
			seen_agents = tuple([p for p in p_obs.players if p.is_self] + [p for p in p_obs.players if not p.is_self])
			seen_foods = tuple(p_obs.foods)
			n_seen_foods = len(seen_foods)
			n_seen_agents = len(seen_agents)
			for idx in range(n_seen_foods):
				if self._agent_center:
					f_row, f_col = self.get_centered_pos(self._players[idx].position, seen_foods[idx].position)
				else:
					f_row, f_col = seen_foods[idx].position
				food_lvl = seen_foods[idx].level
				food_obs[3 * idx] = f_row
				food_obs[3 * idx + 1] = f_col
				food_obs[3 * idx + 2] = 0
				food_obs[3 * idx + food_lvl + 2] = 1

			for idx in range(n_seen_agents):
				if self._agent_center:
					a_row, a_col = self.get_centered_pos(self._players[idx].position, seen_agents[idx].position)
				else:
					a_row, a_col = seen_agents[idx].position
				agent_lvl = seen_agents[idx].level
				agent_obs[3 * idx] = a_row
				agent_obs[3 * idx + 1] = a_col
				agent_obs[3 * idx + 2] = 0
				agent_obs[3 * idx + agent_lvl + 2] = 1

			obs += [food_obs + agent_obs]

			rewards[a_idx] = self._players[a_idx].reward

		return np.array(obs), rewards, done, force_stop, info

	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[np.ndarray, dict[str, Any]]:
		if seed is not None:
			self.seed(seed)
		self.field = np.zeros(self.field_size, np.int32)
		max_food_lvl = self._max_food_lvl if self._max_food_lvl > 0 else sum([p.level for p in self._players])
		self.spawn_food(self._max_spawn_food, max_food_lvl)
		self.spawn_players()
		self._current_step = 0
		self._game_over = False

		obs, _, _, _, info = self.make_gym_obs()
		return obs, info

	def step(self, actions: list[int]) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:

		self._current_step += 1

		for agent in self._players:
			agent.reward = 0

		loading_agents = set()
		agent_moves = defaultdict(list)
		occupied_pos = [tuple(food.position) for food in self._foods if not food.picked]

		for agent, act in zip(self._players, actions):
			action = Action(act)
			if action == Action.NONE:
				agent_moves[agent.position].append(agent)
			elif action == Action.LOAD:
				agent_moves[agent.position].append(agent)
				loading_agents.add(agent)
			else:
				row_delta, col_delta = Direction[action.name].value
				agent_row, agent_col = agent.position
				next_pos = (min(max(0, agent_row + row_delta), self.rows - 1), min(max(0, agent_col + col_delta), self.cols - 1))
				if next_pos in occupied_pos:
					agent_moves[agent.position].append(agent)
				else:
					agent_moves[next_pos].append(agent)

		self.resolve_collisions(agent_moves)
		self.resolve_loading(loading_agents)
		self._game_over = all([food.picked for food in self._foods])
		for p in self._players:
			p.score += p.reward

		return self.make_obs_dqn_array() if self._use_encoding and not self._grid_observation else self.make_gym_obs()

	def render(self) -> np.ndarray | list[np.ndarray] | None:
		if not self._rendering_initialized:
			try:
				from .render import Viewer
				self._render = Viewer((self.rows, self.cols), visible=self._show_viewer)
				self._rendering_initialized = True
			except Exception as e:
				print('Caught exception %s when trying to import Viewer class.' % str(e.args))
				return None

		return self._render.render(self, return_rgb_array=(self.render_mode == 'rgb_array'))

	def close_render(self):
		if self._render is not None:
			self._render.close()
			self._render = None
			self._rendering_initialized = False

	def close(self):
		super().close()
		self.close_render()