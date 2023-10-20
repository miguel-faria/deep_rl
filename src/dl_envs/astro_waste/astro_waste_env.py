#! /usr/bin/env python

import numpy as np
import pylab as p
import random
import yaml
import gymnasium

from pathlib import Path
from enum import IntEnum, Enum
from gymnasium.utils import seeding
from gymnasium.spaces import Discrete, Box
from gymnasium import Env
from typing import List, Tuple, Callable, Any, Dict
from copy import deepcopy
from termcolor import colored
from collections import namedtuple


MOVE_REWARD = 0.0
HOLD_REWARD = -3.0
DELIVER_WASTE = 10
ROOM_CLEAN = 50


class AgentType(IntEnum):
	HUMAN = 0
	ROBOT = 1


class HoldState(IntEnum):
	FREE = 0
	HELD = 1
	DISPOSED = 2


class CellEntity(IntEnum):
	EMPTY = 0
	COUNTER = 1
	ICE = 2
	AGENT = 3
	OBJECT = 4


class Actions(IntEnum):
	UP = 0
	DOWN = 1
	LEFT = 2
	RIGHT = 3
	INTERACT = 4
	STAY = 5


class ActionDirection(Enum):
	UP = (-1, 0)
	DOWN = (1, 0)
	LEFT = (0, -1)
	RIGHT = (0, 1)
	INTERACT = (0, 0)
	STAY = (0, 0)


class ObjectState(object):
	_position: Tuple[int, int]
	_id: str
	_hold_state: int
	_holding_player: 'PlayerState'
	
	def __init__(self, position: Tuple[int, int], obj_id: str, hold_state: int = HoldState.FREE.value, holding_player: 'PlayerState' = None):
		self._position = position
		self._id = obj_id
		self._hold_state = hold_state
		self._holding_player = holding_player
		
	@property
	def position(self) -> Tuple[int, int]:
		return self._position
	
	@property
	def id(self) -> str:
		return self._id
	
	@property
	def hold_state(self) -> int:
		return self._hold_state
	
	@property
	def holding_player(self) -> 'PlayerState':
		return self._holding_player
	
	@position.setter
	def position(self, new_pos: Tuple[int, int]) -> None:
		self._position = new_pos
		
	@hold_state.setter
	def hold_state(self, new_state: int) -> None:
		self._hold_state = new_state

	@holding_player.setter
	def holding_player(self, new_player: 'PlayerState') -> None:
		self._holding_player = new_player

	def deepcopy(self):
		new_obj = ObjectState(self._position, self._id)
		new_obj.hold_state = self._hold_state
		return new_obj

	def __eq__(self, other):
		return isinstance(other, ObjectState) and self._id == other._id and self._position == other._position
	
	def __hash__(self):
		return hash((self._id, self._position))
	
	def __repr__(self):
		return "%s@(%d, %d), held_status: %s" % (self._id, self._position[0], self._position[1], HoldState(self._hold_state).name)
	
	def to_dict(self):
		return {"name": self._id, "position": self._position, "hold_state": self._hold_state, "holding_player": self._holding_player}

	@classmethod
	def from_dict(cls, obj_dict):
		obj_dict = deepcopy(obj_dict)
		return ObjectState(**obj_dict)


class PlayerState(object):
	_position: Tuple[int, int]
	_orientation: Tuple[int, int]
	_id: str
	_agent_type: int
	_held_object: List[ObjectState]
	_reward: float
	
	def __init__(self, pos: Tuple[int, int], orientation: Tuple[int, int], agent_id: str, agent_type: int, held_object: List[ObjectState] = None):
		self._position = pos
		self._orientation = orientation
		self._agent_type = agent_type
		self._id = agent_id
		self._held_object = held_object
		self._reward = 0
		
		if self._held_object is not None:
			for obj in self._held_object:
				assert isinstance(obj, ObjectState)
				assert obj.position == self._position
	
	@property
	def position(self) -> Tuple:
		return self._position
	
	@property
	def orientation(self) -> Tuple:
		return self._orientation
	
	@property
	def agent_type(self) -> int:
		return self._agent_type
	
	@property
	def id(self) -> str:
		return self._id
	
	@property
	def reward(self) -> float:
		return self._reward
	
	@property
	def held_objects(self) -> List[ObjectState]:
		if self._held_object is not None:
			return self._held_object.copy()
		else:
			return self._held_object

	@position.setter
	def position(self, new_pos: Tuple[int, int]) -> None:
		self._position = new_pos
		
	@orientation.setter
	def orientation(self, new_orientation: Tuple[int, int]) -> None:
		self._orientation = new_orientation
	
	@reward.setter
	def reward(self, new_val: float) -> None:
		self._reward = new_val
	
	def hold_object(self, other_obj: ObjectState) -> None:
		assert isinstance(other_obj, ObjectState), "[HOLD OBJECT ERROR] object is not an ObjectState"
		if self._held_object is not None:
			self._held_object.append(other_obj)
		else:
			self._held_object = [other_obj]
			
	def drop_object(self, obj_id: str) -> None:
		assert self.is_holding_object(), "[DROP OBJECT] holding no objects"
		
		for obj in self._held_object:
			if obj.id == obj_id:
				self._held_object.remove(obj)
				return
		
		print(colored('[DROP OBJECT] no object found with id %s' % obj_id, 'yellow'))

	def is_holding_object(self) -> bool:
		if self._held_object is not None:
			return len(self._held_object) > 0
		else:
			return False

	def deepcopy(self):
		held_objs = (None if self._held_object is None else [self._held_object[idx].deepcopy() for idx in range(len(self._held_object))])
		return PlayerState(self._position, self._orientation, self._id, self._agent_type, held_objs)
	
	def __eq__(self, other):
		return (
				isinstance(other, PlayerState)
				and self.position == other.position
				and self.orientation == other.orientation
				and self.held_objects == other.held_objects
		)
	
	def __hash__(self):
		return hash((self.position, self.orientation, self.held_objects))
	
	def __repr__(self):
		return "Agent {} at {} facing {} holding {}".format(self._id, self.position, self.orientation, str(self.held_objects))
	
	def to_dict(self):
		return {
			"position": self.position,
			"orientation": self.orientation,
			"held_object": [self.held_objects[idx].to_dict() for idx in range(len(self._held_object))] if self.held_objects is not None else None,
		}
	
	@staticmethod
	def from_dict(player_dict):
		player_dict = deepcopy(player_dict)
		held_obj = player_dict.get("held_object", None)
		if held_obj is not None:
			player_dict["held_object"] = [ObjectState.from_dict(held_obj[idx]) for idx in range(len(held_obj))]
		return PlayerState(**player_dict)


# noinspection PyUnresolvedReferences
class AstroWasteEnv(Env):
	Observation = namedtuple("Observation", ["field", "players", "objects", "game_finished", "game_timeout", "sight", "current_step"])
	
	def __init__(self, terrain_size: Tuple[int, int], layout: str, max_players: int, slip: bool, max_objects: int, max_steps: int, rnd_seed: int,
				 require_facing: bool = False, layer_obs: bool = False, agent_centered: bool = False, use_encoding: bool = False):
		
		self._np_random, _ = seeding.np_random(rnd_seed)
		self._rows, self._cols = terrain_size
		self._players: List[PlayerState] = []
		self._objects: List[ObjectState] = []
		self._field: np.ndarray = np.zeros((self._rows, self._cols))
		self._max_players = max_players
		self._max_objects = max_objects
		self._n_players = 0
		self._n_objects = 0
		self._room_layout = layout
		self._slip = slip
		self._max_time_steps = max_steps
		self._current_step = 0
		self._n_actions = len(Actions)
		self._slip_prob = 0.0
		self._agent_sight = 0
		self._require_facing = require_facing
		self._render = None
		self._use_layer_obs = layer_obs
		self._agent_centered_obs = agent_centered
		self._use_encoding = use_encoding
		
		self.setup_env()
		self.action_space = Discrete(self._n_actions)
		self.observation_space = gymnasium.spaces.Tuple(tuple([self._get_observation_space()] * self._n_players))
		self.action_space.seed(rnd_seed)
		self.observation_space.seed(rnd_seed)
	
	###########################
	### GETTERS AND SETTERS ###
	###########################
	@property
	def rows(self) -> int:
		return self._rows
	
	@property
	def cols(self) -> int:
		return self._cols
	
	@property
	def players(self) -> List[PlayerState]:
		return self._players
	
	@property
	def objects(self) -> List[ObjectState]:
		return self._objects
	
	@property
	def max_players(self) -> int:
		return self._max_players
	
	@property
	def max_objects(self) -> int:
		return self._max_objects
	
	@property
	def n_players(self) -> int:
		return self._n_players
	
	@property
	def n_objects(self) -> int:
		return self._n_objects
	
	@property
	def layout(self) -> str:
		return self._room_layout
	
	@property
	def max_steps(self) -> int:
		return self._max_time_steps
	
	@property
	def slip(self) -> bool:
		return self._slip
	
	@property
	def require_facing(self) -> bool:
		return self._require_facing
	
	@property
	def field(self) -> np.ndarray:
		return self._field
	
	@slip.setter
	def slip(self, new_val: bool) -> None:
		self._slip = new_val
	
	def seed(self, rng_seed: int):
		self._np_random, _ = seeding.np_random(rng_seed)
		self.action_space.seed(rng_seed)
		self.observation_space.seed(rng_seed)
	
	#######################
	### UTILITY METHODS ###
	#######################
	def _get_observation_space(self) -> Box:
		
		if self._use_layer_obs:
			if self._agent_centered_obs:
				# grid observation space
				grid_shape = (1 + 2 * self._agent_sight, 1 + 2 * self._agent_sight)
				
				# agents layer: agent levels
				agents_min = np.zeros(grid_shape, dtype=np.int32)
				agents_max = np.ones(grid_shape, dtype=np.int32)
				
				# foods layer: foods level
				objs_min = np.zeros(grid_shape, dtype=np.int32)
				objs_max = np.ones(grid_shape, dtype=np.int32)
				
				# access layer: i the cell available
				occupancy_min = np.zeros(grid_shape, dtype=np.int32)
				occupancy_max = np.ones(grid_shape, dtype=np.int32)
				
				# total layer
				min_obs = np.stack([agents_min, objs_min, occupancy_min])
				max_obs = np.stack([agents_max, objs_max, occupancy_max])
			
			else:
				# grid observation space
				grid_shape = (self._rows, self._cols)
				
				# agents layer
				agents_min = np.zeros(grid_shape, dtype=np.int32)
				agents_max = np.ones(grid_shape, dtype=np.int32)
				
				# objects layer
				objs_min = np.zeros(grid_shape, dtype=np.int32)
				objs_max = np.ones(grid_shape, dtype=np.int32)
				
				# occupancy layer
				occupancy_min = np.zeros(grid_shape, dtype=np.int32)
				occupancy_max = np.ones(grid_shape, dtype=np.int32)
				
				# acting agent layer
				acting_agent_min = np.zeros(grid_shape, dtype=np.int32)
				acting_agent_max = np.ones(grid_shape, dtype=np.int32)
				
				# total layer
				min_obs = np.stack([agents_min, objs_min, occupancy_min, acting_agent_min])
				max_obs = np.stack([agents_max, objs_max, occupancy_max, acting_agent_max])
		else:
			if self._use_encoding:
				min_obs = [-1, -1, 0, 0, 0] * self._n_players + [-1, -1, *[0] * len(HoldState), *[0] * self._n_players] * self._n_objects
				max_obs = ([self._rows - 1, self._cols - 1, 1, 1, 1] * self._n_players +
						   [self._rows - 1, self._cols - 1, *[1] * len(HoldState), *[1] * self._n_players] * self._n_objects)
			else:
				min_obs = [-1, -1, -1, -1, 0] * self._n_players + [-1, -1, 0, 0] * self._n_objects
				max_obs = [self._rows - 1, self._cols - 1, 1, 1, 1] * self._n_players + [self._rows - 1, self._cols - 1, 2, self._n_players - 1] * self._n_objects
		
		return Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)
	
	def get_filled_field(self) -> np.ndarray:
		
		field = self._field.copy()
		for agent in self._players:
			field[agent.position] = CellEntity.AGENT
			
		for obj in self._objects:
			field[obj.position] = CellEntity.OBJECT
		
		return field
	
	def setup_env(self) -> None:
		
		config_filepath = Path(__file__).parent.absolute() / 'data' / 'configs' / 'layouts' / (self._room_layout + '.yaml')
		with open(config_filepath) as config_file:
			config_data = yaml.safe_load(config_file)
		field_data = config_data['field']
		players_data = config_data['players']
		objects_data = config_data['objects']
		self._slip_prob = float(config_data['slip_prob'])
		config_sight = float(config_data['sight'])
		self._agent_sight = config_sight if config_sight > 0 else min(self._rows, self._cols)
		
		for row in range(self._rows):
			for col in range(self._cols):
				cell_val = field_data[row][col]
				if cell_val == ' ':
					pass
				elif cell_val == 'X':
					self._field[row, col] = CellEntity.COUNTER
				elif cell_val == 'I':
					self._field[row, col] = CellEntity.ICE
				elif cell_val == 'O':
					self.add_object((row, col), objects_data[self._n_objects])
					self._field[row, col] = CellEntity.COUNTER
				elif cell_val.isdigit():
					nxt_player_data = players_data[self._n_players]
					self.add_player((row, col), tuple(nxt_player_data['orientation']), nxt_player_data['id'],
									AgentType[nxt_player_data['type'].upper()].value)
				else:
					print(colored("[SETUP_ENV] Cell value %s not recognized, considering empty cell" % cell_val, 'yellow'))
					continue
	
	@staticmethod
	def encode_orientation(orientation: Tuple) -> Tuple[int, int]:
		
		if orientation == ActionDirection.UP.value:
			return 0, 0
		elif orientation == ActionDirection.DOWN.value:
			return 0, 1
		elif orientation == ActionDirection.LEFT.value:
			return 1, 0
		elif orientation == ActionDirection.RIGHT.value:
			return 1, 1
		else:
			return 0, 0
	
	@staticmethod
	def get_adj_pos(position: Tuple, ignore_diag: bool = True) -> List[Tuple[int, int]]:
		if ignore_diag:
			return [(position[0] - 1, position[1]), (position[0] + 1, position[1]), (position[0], position[1] - 1), (position[0], position[1] - 1)]
		
		else:
			return [(position[0] - 1, position[1]), (position[0] + 1, position[1]), (position[0], position[1] - 1),
					(position[0], position[1] + 1), (position[0] - 1, position[1] - 1), (position[0] + 1, position[1] - 1),
					(position[0] - 1, position[1] + 1), (position[0] + 1, position[1] + 1)]
		
	def disposed_objects(self) -> List[ObjectState]:
		return [obj for obj in self._objects if obj.hold_state == HoldState.DISPOSED]
	
	def is_game_finished(self) -> bool:
		return (self._n_objects - len(self.disposed_objects())) == 0
	
	def is_game_timedout(self) -> bool:
		return self._current_step >= self.max_steps
	
	def is_over(self) -> bool:
		return self.is_game_finished() or self.is_game_timedout()
	
	@staticmethod
	def are_facing(agent_1: PlayerState, agent_2: PlayerState) -> bool:
		return (agent_1.orientation[0] + agent_2.orientation[0]) == 0 and (agent_1.orientation[1] + agent_2.orientation[1]) == 0
	
	def move_ice(self, move_agent: PlayerState, next_position: Tuple) -> Tuple:
		
		agent_pos = move_agent.position
		right_move = (next_position[0] - agent_pos[0], next_position[1] - agent_pos[1])
		wrong_moves = [direction.value for direction in ActionDirection if direction.value != right_move and direction.value != (0, 0)]
		n_wrong_moves = len(wrong_moves)
		moves_prob = np.array([1 - self._slip_prob] + [self._slip_prob / n_wrong_moves] * n_wrong_moves)
		possible_positions = ([next_position] + [(max(min(wrong_move[0] + agent_pos[0], self._rows), 0), max(min(wrong_move[1] + agent_pos[1], self.cols), 0))
												 for wrong_move in wrong_moves])
		return possible_positions[self._np_random.choice(range(len(possible_positions)), p=moves_prob)]
	
	def free_pos(self, pos: Tuple[int, int]) -> bool:
	
		for player in self._players:
			if player.position == pos:
				return False
	
		for obj in self._objects:
			if obj.position == pos:
				return False
	
		return True
	
	def get_agent_facing(self, player: PlayerState) -> PlayerState:
		
		facing_pos = (player.position[0] + player.orientation[0], player.position[1] + player.orientation[1])
		for agent in self._players:
			if agent.position == facing_pos:
				return agent
			
		return  None
	
	####################
	### MAIN METHODS ###
	####################
	def get_env_log(self) -> str:
		
		env_log = 'Environment state:\nPlayer states:\n'
		for player in self._players:
			env_log += '\t- ' + str(player) + '\n'
		
		env_log += 'Object states:\n'
		for obj in self._objects:
			if obj.hold_state != HoldState.HELD:
				env_log += '\t- ' + str(obj) + '\n'
		
		env_log += 'Field layout: %s\n' % str(self._field)
		env_log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._current_step, self.is_game_finished(), self.is_game_timedout())
		
		return env_log
	
	def create_observation(self) -> Observation:
		return self.Observation(field=self.field,
								players=self.players,
								objects=self.objects,
								game_finished=self.is_game_finished(),
								game_timeout=self.is_game_timedout(),
								sight=self._agent_sight,
								current_step=self._current_step)
	
	def add_player(self, position: Tuple, orientation: Tuple = (-1, 0), agent_id: str = 'human', agent_type: int = AgentType.HUMAN,
				   held_objs: List[ObjectState] = None) -> bool:
		
		if self._n_players < self._max_players:
			self._players.append(PlayerState(position, orientation, agent_id, agent_type, held_objs))
			self._n_players += 1
			return True
		else:
			print(colored('[ADD_PLAYER] Max number players (%d) already reached, cannot add a new one.' % self._max_players, 'yellow'))
			return False
	
	def add_object(self, position: Tuple, obj_id: str = 'ball') -> bool:
		
		if self._n_objects < self._max_objects:
			self._objects.append(ObjectState(position, obj_id))
			self._n_objects += 1
			return True
		else:
			print(colored('[ADD_OBJECT] Max number of objects (%d) already reached, cannot add a new one.' % self._max_objects, 'yellow'))
			return False
	
	def remove_player(self, player_id: str) -> bool:
		
		for player in self._players:
			if player.id == player_id:
				if player.is_holding_object():
					if len(player.held_objects) > 1:
						obj_idx = 0
						adj_pos = self.get_adj_pos(player.position, False)
						n_adj_pos = len(adj_pos)
						for obj in player.held_objects:
							obj.hold_state = HoldState.FREE
							if obj_idx < n_adj_pos:
								obj.position = adj_pos[obj_idx]
							else:
								obj.position = player.position
							obj_idx += 1
					else:
						obj = player.held_objects[0]
						obj.hold_state = HoldState.FREE.value
				self._players.remove(player)
				self._n_players -= 1
				return True
		
		print(colored('Player with id %s not found in list' % player_id, 'yellow'))
		return False
	
	def remove_object(self, obj_id: str) -> bool:
		
		for obj in self._objects:
			if obj.id == obj_id:
				if obj.hold_state == HoldState.HELD or obj.hold_state == HoldState.DISPOSED:
					obj.holding_player.held_objects.remove(obj)
				self._objects.remove(obj)
				self._n_objects -= 1
				return True
		
		print(colored('Object with id %s not found in list' % obj_id, 'yellow'))
		return False
	
	def make_obs(self) -> np.ndarray:
		if self._use_layer_obs:
			return self.make_obs_grid()
		elif self._use_encoding:
			return self.make_obs_dqn()
		else:
			return self.make_obs_array()
	
	def make_obs_array(self) -> np.ndarray:
		state = []
		
		for player in self._players:
			player_state = [player.position, player.orientation, int(player.is_holding_object())]
			for other_player in self._players:
				if other_player.id != player.id:
					player_state.append(other_player.position)
					player_state.append(other_player.orientation)
					player_state.append(int(other_player.is_holding_object()))
			for obj in self._objects:
				if obj.hold_state == HoldState.DISPOSED:  # When disposed, object position is a virtual trash bin at (-1, -1)
					player_state.append((-1, -1))
				else:
					player_state.append(obj.position)
				player_state.append(obj.hold_state)
				if obj.hold_state == HoldState.FREE:  # If the object is not held, the agent holding it is 0 (None)
					player_state.append(0)
				else:
					player_state.append(self._players.index(obj.holding_player) + 1)
			
			state.append(player_state)
		
		return np.array(state)
	
	def make_obs_grid(self) -> np.ndarray:
		
		if self._agent_centered_obs:
			layers_size = (self._rows + 2 * self._agent_sight, self._cols + 2 * self._agent_sight)
			agent_layer = np.zeros(layers_size, dtype=np.int32)
			obj_layer = np.zeros(layers_size, dtype=np.int32)
			occupancy_layer = np.ones(layers_size, dtype=np.int32)
			occupancy_layer[:self._agent_sight, :] = 0
			occupancy_layer[-self._agent_sight:, :] = 0
			occupancy_layer[:, :self._agent_sight] = 0
			occupancy_layer[:, -self._agent_sight:] = 0
			
			for agent in self._players:
				pos = agent.position
				agent_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				occupancy_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 0
				
			for obj in self._objects:
				pos = obj.position
				obj_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 1
				occupancy_layer[pos[0] + self._agent_sight, pos[1] + self._agent_sight] = 0
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row + self._agent_sight, col + self._agent_sight] = 0
			
			obs = np.stack([agent_layer, obj_layer, occupancy_layer])
			padding = 2 * self._agent_sight + 1
			
			return np.array([obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding] for a in self._players])
		
		else:
			layers_size = (self._rows, self._cols)
			agent_layer = np.zeros(layers_size, dtype=np.int32)
			obj_layer = np.zeros(layers_size, dtype=np.int32)
			occupancy_layer = np.ones(layers_size, dtype=np.int32)
			acting_layer = np.zeros((self._n_players, *layers_size), dtype=np.int32)
			
			for agent_idx in range(self._n_players):
				pos = self._players[agent_idx].position
				agent_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
				acting_layer[agent_idx, pos[0], pos[1]] = 1
			
			for obj in self._objects:
				pos = obj.position
				obj_layer[pos[0], pos[1]] = 1
				occupancy_layer[pos[0], pos[1]] = 0
			
			for row in range(self._rows):
				for col in range(self._cols):
					if self._field[row, col] == CellEntity.COUNTER:
						occupancy_layer[row, col] = 0
			
			return np.array([np.stack([agent_layer, obj_layer, occupancy_layer, acting_layer[idx]]) for idx in range(self._n_players)])
	
	def make_obs_dqn(self) -> np.ndarray:
		state = []
		
		for player in self._players:
			player_state = [player.position, self.encode_orientation(player.orientation), int(player.is_holding_object())]
			for other_player in self._players:
				if other_player.id != player.id:
					player_state.append(other_player.position)
					player_state.append(self.encode_orientation(other_player.orientation))
					player_state.append(int(other_player.is_holding_object()))
			for obj in self._objects:
				if obj.hold_state == HoldState.DISPOSED:  # When disposed, object position is a virtual trash bin at (-1, -1)
					player_state.append((-1, -1))
				else:
					player_state.append(obj.position)
				hold_state = [0] * len(HoldState)
				hold_state[obj.hold_state] = 1
				player_state.append(*hold_state)
				if obj.hold_state == HoldState.FREE:  # If the object is not held, the agent holding it is 0 (None)
					player_state.append([0] * self._n_players)
				else:
					hold_player = [0] * self._n_players
					hold_player[self._players.index(obj.holding_player)] = 1
					player_state.append(*hold_player)
			
			state.append(player_state)
		
		return np.array(state)
	
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		
		if seed is not None:
			self.seed(seed)
		
		self._current_step = 0
		self._n_players = 0
		self._n_objects = 0
		self._players: List[PlayerState] = []
		self._objects: List[ObjectState] = []
		self._field: np.ndarray = np.zeros((self._rows, self._cols))
		self.setup_env()
		
		return self.make_obs(), {}
	
	def step(self, actions: List[int]) -> tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:
		
		self._current_step += 1
		old_positions = []
		for player in self._players:
			player.reward = MOVE_REWARD
			old_positions += [player.position]
		
		# Process action list
		new_positions = []
		slip_agents = []
		agents_disposed_waste = []
		for act in actions:
			act_idx = actions.index(act)
			acting_player = self._players[act_idx]
			act_direction = ActionDirection[Actions(act).name].value
			if act != Actions.INTERACT and act != Actions.STAY:
				acting_player.orientation = act_direction
			next_pos = (max(min(acting_player.position[0] + act_direction[0], self._rows), 0),
						max(min(acting_player.position[1] + act_direction[1], self.cols), 0))
			if self._slip and self._field[acting_player.position] == CellEntity.ICE:
				new_positions.append(self.move_ice(acting_player, next_pos))
				slip_agents.append(acting_player.id)
			else:
				new_positions.append(next_pos)
			
			# Handle INTERACT action is only necessary for human agents
			if act == Actions.INTERACT and acting_player.agent_type == AgentType.HUMAN:
				facing_pos = (acting_player.position[0] + acting_player.orientation[0], acting_player.position[1] + acting_player.orientation[1])
				agent_facing = self.get_agent_facing(acting_player)
				if acting_player.is_holding_object():
					if agent_facing is not None: # facing an agent
						agent_idx = self._players.index(agent_facing)
						agent_action = actions[agent_idx]
						agent_type = agent_facing.agent_type
						if agent_type == AgentType.ROBOT and (agent_action == Actions.STAY or agent_action == Actions.INTERACT): # check if the agent is a robot and is not trying to move
							if self.require_facing and not self.are_facing(acting_player, agent_facing):
								continue
							# Place object in robot
							place_obj = acting_player.held_objects[0]
							acting_player.drop_object(place_obj.id)
							place_obj.hold_state = HoldState.DISPOSED
							place_obj.holding_player = agent_facing
							place_obj.position = (-1, -1)
							agent_facing.hold_object(place_obj)
							agents_disposed_waste.append(acting_player)
							agents_disposed_waste.append(agent_facing)
					else:
						# Drop object to the field
						dropped_obj = acting_player.held_objects[0]
						if dropped_obj.hold_state == HoldState.HELD and self.free_pos(facing_pos):
							acting_player.drop_object(dropped_obj.id)
							dropped_obj.position = facing_pos
							dropped_obj.hold_state = HoldState.FREE
							dropped_obj.holding_player = None
						
				else:
					if agent_facing is None:
						# Pick object from counter or floor
						for obj in self._objects:
							if obj.position == facing_pos and obj.hold_state == HoldState.FREE:
								pick_obj = obj
								pick_obj.position = acting_player.position
								pick_obj.hold_state = HoldState.HELD
								pick_obj.holding_player = acting_player
								acting_player.hold_object(pick_obj)
							
		# Handle movement and collisions
		# Check for collisions (more than one agent moving to same position or two agents switching position)
		can_move = []
		for idx in range(self._n_players):
			curr_pos = old_positions[idx]
			next_pos = new_positions[idx]
			add_move = True
			for idx2 in range(self._n_players):
				if idx2 == idx:
					continue
				if new_positions[idx2] == next_pos or (curr_pos == new_positions[idx2] and next_pos == old_positions[idx2]):
					add_move = False
					break
			if add_move:
				can_move.append(idx)
				
		# Update position for agents with valid movements
		for idx in can_move:
			moving_player = self._players[idx]
			old_pos = old_positions[idx]
			next_pos = new_positions[idx]
			if old_pos != next_pos and self._field[next_pos] == CellEntity.EMPTY:
				moving_player.position = next_pos
				if moving_player.is_holding_object():
					for obj in moving_player.held_objects:
						if obj.hold_state != HoldState.DISPOSED:
							obj.position = next_pos
		
		for player in self._players:
			if self.is_game_finished():
				player.reward = ROOM_CLEAN
			elif player in agents_disposed_waste:
				player.reward = DELIVER_WASTE
			else:
				facing_agent = self.get_agent_facing(player)
				if (facing_agent is not None and
						((player.agent_type == AgentType.HUMAN and facing_agent.agent_type == AgentType.ROBOT and player.is_holding_object()) or
						 (player.agent_type == AgentType.ROBOT and facing_agent.agent_type == AgentType.HUMAN and facing_agent.is_holding_object()))):
					player.reward = DELIVER_WASTE / self._max_time_steps
		
		finished = self.is_game_finished()
		rewards = np.array([player.reward for player in self._players])
		timeout = self.is_game_timedout()
		return self.make_obs(), rewards, finished, timeout, {'agents_slipped': slip_agents}
	
	def render(self) -> np.ndarray:
		pass