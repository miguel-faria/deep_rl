#! /usr/bin/env python

from .agent import Agent
from termcolor import colored


class TargetAgent(Agent):
	
	_target: str
	
	def __init__(self, a_id: str, agent_type: int, rank: int = 0):
		
		super().__init__(a_id, agent_type, rank)
		self._target = ''
		
	@property
	def target(self) -> str:
		return self._target
	
	@target.setter
	def target(self, new_target: str) -> None:
		self._target = new_target
		
	def get_reward(self, raw_reward: float, **extra_info) -> float:
		env = extra_info.get("env")
		if self._target == '':
			print(colored('No target prey defined, defaulting to base Agent reward', 'yellow'))
			return super().get_reward(raw_reward, **extra_info)
		if env is None:
			print(colored('No environment provided, defaulting to base Agent reward', 'yellow'))
			return super().get_reward(raw_reward, **extra_info)
		prey = env.agents[self._target]
		hunter = env.agents[self._id]
		prey_adj_pos = [(prey.pos[0] - 1, prey.pos[1]), (prey.pos[0] + 1, prey.pos[1]), (prey.pos[0], prey.pos[1] - 1), (prey.pos[0], prey.pos[1] + 1)]
		
		if prey.alive and raw_reward > 0 and hunter.pos not in prey_adj_pos:
			return 0
		else:
			return raw_reward