#! /usr/bin/env python

import numpy as np

from .astro_waste_env import Actions, CellEntity, ActionDirection, AstroWasteEnv, PlayerState, ObjectState, AgentType
from typing import List, Tuple, Dict
from enum import IntEnum


class WasteStatus(IntEnum):
	DROPPED = 0
	PICKED = 1
	DISPOSED = 2


class HumanStatus(IntEnum):
	HANDS_FREE = 0
	WASTE_PICKED = 1


class PosNode(object):
	
	_pos: Tuple[int, int]
	_cost: int
	_parent: 'PosNode'
	
	def __init__(self, pos: Tuple[int, int], cost: int, parent_node: 'PosNode'):
		self._pos = pos
		self._cost = cost
		self._parent = parent_node
	
	@property
	def pos(self) -> Tuple[int, int]:
		return self._pos
	
	@property
	def cost(self) -> int:
		return self._cost
	
	@property
	def parent(self) -> 'PosNode':
		return self._parent
	
	@cost.setter
	def cost(self, new_cost: int) -> None:
		self._cost = new_cost
		
	def __str__(self):
		return 'Pos (%d, %d) with cost: %d' % (*self._pos, self._cost)


def get_adj_pos(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
	return [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]


class GreedyHumanAgent(object):
	
	_pos: Tuple[int, int]
	_orientation: Tuple[int, int]
	_agent_id: str
	_status: HumanStatus
	_nxt_waste_idx: int
	_waste_pos: Dict[int, Tuple[int, int]]
	_waste_order: List[int] = None
	_rng_gen: np.random.Generator
	_map_adjacencies: Dict[Tuple[int, int], List[Tuple[int, int]]]
	
	def __init__(self, pos_init: Tuple[int, int], orient_init: Tuple[int, int], agent_id: str, objs_pos: Dict[int, Tuple[int, int]],
				 rng_seed: int, field: np.ndarray):
		
		self._pos = pos_init
		self._orientation = orient_init
		self._agent_id = agent_id
		self._status = HumanStatus.HANDS_FREE
		self._nxt_waste_idx = -1
		self._waste_pos = objs_pos.copy()
		self._rng_gen = np.random.default_rng(rng_seed)
		
		# Create adjacency map
		rows, cols = field.shape
		free_pos = [(row, col) for row in range(rows) for col in range(cols) if field[row, col] == CellEntity.EMPTY or field[row, col] == CellEntity.ICE]
		free_pos.sort()
		self._map_adjacencies = {}
		for pos in free_pos:
			adj_key = (pos[0], pos[1])
			adjacencies = []
			adjs_pos = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)]
			for a_pos in adjs_pos:
				if a_pos in free_pos:
					adjacencies += [(a_pos[0], a_pos[1])]
			self._map_adjacencies[adj_key] = adjacencies
	
	@property
	def pos(self) -> Tuple[int, int]:
		return self._pos
	
	@property
	def orientation(self) -> Tuple[int, int]:
		return self._orientation
	
	@property
	def agent_id(self) -> str:
		return self._agent_id
	
	@property
	def status(self) -> HumanStatus:
		return self._status
	
	@property
	def next_waste_idx(self) -> int:
		return self._nxt_waste_idx

	@property
	def waste_order(self) -> List[int]:
		return self._waste_order

	@property
	def waste_pos(self) -> Dict[int, Tuple[int, int]]:
		return self._waste_pos

	@pos.setter
	def pos(self, new_pos: Tuple[int, int]) -> None:
		self._pos = new_pos
	
	@status.setter
	def status(self, new_status: HumanStatus) -> None:
		self._status = new_status
	
	@next_waste_idx.setter
	def next_waste_idx(self, nxt_obj: int) -> None:
		self._nxt_waste_idx = nxt_obj
		
	@waste_order.setter
	def waste_order(self, new_order: List[int]) -> None:
		self._waste_order = new_order.copy()
	
	def __repr__(self):
		return 'Greedy agent {} at {} facing {} with objective {} and with {}'.format(self._agent_id, self._pos, self._orientation,
																					  self._waste_pos[self._nxt_waste_idx], HumanStatus(self._status).name)
		
	def reset(self, waste_order: List[int], objs_pos: Dict[int, Tuple[int, int]]):
		self._nxt_waste_idx = -1
		self._status = HumanStatus.HANDS_FREE
		self._waste_order = waste_order.copy()
		self._waste_pos = objs_pos.copy()
		
	def act(self, obs: AstroWasteEnv.Observation) -> int:
		
		def are_facing(h_or: Tuple[int, int], r_or: Tuple[int, int]) -> bool:
			return (h_or[0] + r_or[0]) == 0 and (h_or[1] + r_or[1]) == 0
		
		self_agent = [agent for agent in obs.players if agent.id == self._agent_id][0]
		robot = [agent for agent in obs.players if agent.agent_type == AgentType.ROBOT][0]
		self._pos = self_agent.position
		self._orientation = self_agent.orientation
		self._status = HumanStatus.WASTE_PICKED if self_agent.is_holding_object() else HumanStatus.HANDS_FREE
		robot_pos = robot.position
		robot_or = robot.orientation
		
		if self._nxt_waste_idx < 0:
			self._nxt_waste_idx = self._waste_order.pop(0)
			
		if self._status == HumanStatus.HANDS_FREE:
			nxt_waste = self._waste_pos[self._nxt_waste_idx]
			found_waste = False
			for obj in obs.objects:
				waste_pos = obj.position
				if waste_pos == nxt_waste:
					found_waste = True
					break
				else:
					waste_adj_pos = [(waste_pos[0] - 1, waste_pos[1]), (waste_pos[0] + 1, waste_pos[1]),
									 (waste_pos[0], waste_pos[1] - 1), (waste_pos[0], waste_pos[1] + 1)]
					if nxt_waste in waste_adj_pos:
						self._waste_pos[self._nxt_waste_idx] = waste_pos
						nxt_waste = waste_pos
						found_waste = True
						break
			
			if not found_waste:
				self._nxt_waste_idx = self._waste_order.pop(0)
				nxt_waste = self._waste_pos[self._nxt_waste_idx]
			
			human_adj_pos = get_adj_pos(self._pos)
			if nxt_waste in human_adj_pos and nxt_waste[0] == (self._pos[0] + self._orientation[0]) and nxt_waste[1] == (self._pos[1] + self._orientation[1]):
				return int(Actions.INTERACT)
			
			else:
				return int(self.move_to_position(nxt_waste))
		
		else:
			self._waste_pos[self._nxt_waste_idx] = self._pos
			
			if robot_pos == (self._pos[0] + self._orientation[0], self._pos[1] + self._orientation[1]):
				if are_facing(self._orientation, robot_or):
					return int(Actions.INTERACT)
				else:
					return int(Actions.STAY)
			
			else:
				return int(self.move_to_position(robot_pos))
	
	def expand_pos(self, start_node: PosNode, objective_pos: Tuple[int]) -> Tuple[int]:
		
		node_pos = start_node.pos
		cost = start_node.cost
		
		pos_adj = get_adj_pos(node_pos)
		if objective_pos in pos_adj:
			return objective_pos
		
		else:
			seen_nodes = [start_node]
			nodes_visit = []
			for nxt_pos in self._map_adjacencies[node_pos]:
				nodes_visit += [PosNode(nxt_pos, cost + 1, None)]
				seen_nodes += [nodes_visit[-1]]
				
			nxt_node = None
			done = False
			while not done:
				nxt_node = nodes_visit.pop(0)
				nxt_node_adj = get_adj_pos(nxt_node.pos)
				if objective_pos in nxt_node_adj:
					done = True
				else:
					seen_pos = [node.pos for node in seen_nodes]
					for pos in self._map_adjacencies[nxt_node.pos]:
						if pos not in seen_pos:
							nodes_visit += [PosNode(pos, nxt_node.cost + 1, nxt_node)]
							seen_nodes += [nodes_visit[-1]]
							
			while nxt_node.parent is not None:
				nxt_node = nxt_node.parent
			
			return nxt_node.pos
	
	def move_to_position(self, objective_pos: Tuple[int]) -> int:
		d_row = objective_pos[0] - self._pos[0]
		d_col = objective_pos[1] - self._pos[1]

		if d_row < 0:
			if d_col < 0:
				action = self._rng_gen.choice([Actions.UP, Actions.LEFT])
			elif d_col > 0:
				action = self._rng_gen.choice([Actions.UP, Actions.RIGHT])
			else:
				action = Actions.UP
		elif d_row > 0:
			if d_col < 0:
				action = self._rng_gen.choice([Actions.DOWN, Actions.LEFT])
			elif d_col > 0:
				action = self._rng_gen.choice([Actions.DOWN, Actions.RIGHT])
			else:
				action = Actions.DOWN
		else:
			if d_col < 0:
				action = Actions.LEFT
			else:
				action = Actions.RIGHT

		nxt_pos = (self._pos[0] + ActionDirection[Actions(action).name].value[0], self._pos[1] + ActionDirection[Actions(action).name].value[1])
		if nxt_pos in self._map_adjacencies[self._pos] or nxt_pos == objective_pos:
			return action
		else:
			best_pos = self.expand_pos(PosNode(self._pos, 1, None), objective_pos)
			mv_direction = (best_pos[0] - self._pos[0], best_pos[1] - self._pos[1])
			if mv_direction[0] == -1 and mv_direction[1] == 0:
				return Actions.UP
			elif mv_direction[0] == 1 and mv_direction[1] == 0:
				return Actions.DOWN
			elif mv_direction[0] == 0 and mv_direction[1] == -1:
				return Actions.LEFT
			else:
				return Actions.RIGHT
