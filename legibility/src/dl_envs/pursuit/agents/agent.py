#! /usr/bin/env python

import numpy as np

from typing import Tuple
from enum import IntEnum, Enum


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


class AgentType(IntEnum):
	HUNTER = 1
	PREY = 2


class Agent(object):
	_id: str
	_rank: int
	_agent_type: AgentType
	_pos: Tuple[int, int]
	_alive: bool
	_np_random: np.random.Generator
	
	def __init__(self, a_id: str, agent_type: int, rank: int = 0, rng_seed: int = 123456789):
		self._id = a_id
		self._agent_type = AgentType(agent_type)
		self._pos = (-1, -1)
		self._rank = rank
		self._alive = False
		self._team = ''
		self._np_random = np.random.default_rng(rng_seed)
	
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
	
	@property
	def team(self) -> str:
		return self._team
	
	@team.setter
	def team(self, new_team) -> None:
		self._team = new_team
	
	@pos.setter
	def pos(self, new_pos: Tuple[int, int]) -> None:
		self._pos = new_pos
	
	@alive.setter
	def alive(self, new_alive: bool) -> None:
		self._alive = new_alive
	
	def get_reward(self, raw_reward: float, **extra_info) -> float:
		return raw_reward
	
	def act(self, env) -> int:
		return Action.STAY