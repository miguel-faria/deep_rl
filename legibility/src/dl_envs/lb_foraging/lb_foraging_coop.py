#! /usr/bin/env python

import gymnasium
import numpy as np

from collections import namedtuple
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv, CellEntity, Food
from typing import List, Tuple, Dict, Optional
from gymnasium.envs import Any
from gymnasium.spaces import Box, Space, MultiBinary


MOVE_REWARD = 0.000
REWARD_PICK = 5.0
ADJ_FOOD_REWARD = 0.002
WRONG_PICK = 0.0
RNG_SEED = 27062023


class LimitedCOOPLBForaging(LBForagingEnv):

	_foods_pos: List[Tuple[int, int]]
	_n_food_spawn: int = 0
	_pos_food_spawn: List[Tuple[int, int]] = None

	def __init__(self, players: int, max_player_level: int, field_size: Tuple[int, int], max_food: int, sight: int, max_episode_steps: int,
	             force_coop: bool, food_level: int, rng_seed: int, foods_pos: List = None, render_mode: List = None, use_encoding: bool = False,
	             agent_center: bool = False, grid_observation: bool = False, use_render: bool = False):

		super().__init__(players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop, max_food_lvl=food_level,
		                 render_mode=render_mode, use_encoding=use_encoding, agent_center=agent_center, grid_observation=grid_observation, use_render=use_render)

		self.reward_range = (MOVE_REWARD, REWARD_PICK)
		self.seed(rng_seed)
		if foods_pos is None:
			self._foods_pos = []
			n_foods = len(self._foods_pos)
			while n_foods < max_food:
				tmp_pos_idx = self.np_random.choice(field_size[0] * field_size[1])
				tmp_pos = (tmp_pos_idx // field_size[1], tmp_pos_idx % field_size[1])
				neighbours = [(x, y) for x in range(max(tmp_pos[0] - 1, 0), min(tmp_pos[0] + 1, field_size[0]) + 1)
				              for y in range(max(tmp_pos[1] - 1, 0), min(tmp_pos[1] + 1, field_size[1]) + 1)]
				can_add = True
				for pos in neighbours:
					if pos in self._foods_pos:
						can_add = False
						break
				if can_add:
					self._foods_pos += [tuple(tmp_pos)]
					n_foods = len(self._foods_pos)
		else:
			self._foods_pos = foods_pos.copy()

	def spawn_players(self, player_levels: List = None, player_pos: List = None):
		if player_pos is None:
			super().spawn_players(player_levels)
		else:
			player_count = 0
			for player in self.players:
				player_idx = self.players.index(player)
				row, col = player_pos[player_idx]
				player_lvl = player_levels[player_idx] if player_levels is not None else self.np_random.integers(1, self._max_player_level, endpoint=True)
				player.setup((row, col), player_lvl, self.field_size, player_count + 1)
				self.field[row, col] = CellEntity.AGENT
				player_count += 1

	def spawn_food(self, max_food: int, max_level: int):
		min_level = max_level if self._force_coop else 1
		self.field = np.zeros(self.field_size, np.int32)
		foods_spawned = []
		food_count = 0

		if self._n_food_spawn < 1:
			self._n_food_spawn = max_food

		if max_food < self._max_spawn_food:
			# Randomly pick food items to spawn
			if self._pos_food_spawn is None:
				if max_food > 1:
					pos_spawn = [self.food_pos[idx] for idx in sorted(self.np_random.choice(range(self._max_spawn_food), size=max_food, replace=False))]
				else:
					pos_spawn = [self.food_pos[self.np_random.choice(range(self._max_spawn_food))]]
				self._pos_food_spawn = pos_spawn.copy()
			for pos in self._pos_food_spawn:
				row, col = pos
				if self.field[row, col] == CellEntity.EMPTY:
					food_lvl = min_level if min_level == max_level else self.np_random.integers(min_level, max_level + 1)
					new_food = Food()
					new_food.setup(pos, food_lvl, food_count + 1)
					self.field[row, col] = CellEntity.FOOD
					food_count += 1
					foods_spawned.append(new_food)

		else:
			if self._pos_food_spawn is None:
				self._pos_food_spawn = []
			# Spawn all food items
			for pos in self._foods_pos:
				row, col = pos
				if self.field[row, col] == CellEntity.EMPTY:
					food_lvl = min_level if min_level == max_level else self._np_random.integers(min_level, max_level + 1)
					new_food = Food()
					new_food.setup(pos, food_lvl, food_count + 1)
					self.field[row, col] = CellEntity.FOOD
					food_count += 1
					foods_spawned.append(new_food)

		self._food_spawned = food_count
		self._foods = foods_spawned.copy()

	def make_obs_array(self) -> np.ndarray:

		obs = np.array([np.zeros(self.observation_space[0].shape, dtype=np.int32)] * self._n_agents)
		raw_obs = [self.make_obs(p) for p in self._players]

		for idx in range(self._n_agents):
			p_obs = raw_obs[idx]
			player = self._players[idx]
			seen_agents = tuple([p for p in p_obs.players if p.is_self] + [p for p in p_obs.players if not p.is_self])
			seen_foods = tuple(p_obs.foods)
			n_seen_agents = len(seen_agents)
			seen_foods_pos = [food.position for food in seen_foods]

			for food_pos in self._foods_pos:
				if food_pos in seen_foods_pos:
					if self._agent_center:
						f_row, f_col = self.get_centered_pos(player.position, food_pos)
					else:
						f_row, f_col = food_pos
					food_lvl = seen_foods[seen_foods_pos.index(food_pos)].level
					i = self._foods_pos.index(food_pos)
					obs[idx][3 * i] = f_row
					obs[idx][3 * i + 1] = f_col
					obs[idx][3 * i + 2] = food_lvl

			for i in range(n_seen_agents):
				if self._agent_center:
					p_row, p_col = self.get_centered_pos(player.position, seen_agents[i].position)
				else:
					p_row, p_col = seen_agents[i].position
				obs[idx][3 * self._max_spawn_food + 3 * i] = p_row
				obs[idx][3 * self._max_spawn_food + 3 * i + 1] = p_col
				obs[idx][3 * self._max_spawn_food + 3 * i + 2] = seen_agents[i].level

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
			n_seen_agents = len(seen_agents)
			for food in seen_foods:
				if self._agent_center:
					f_row, f_col = self.get_centered_pos(self._players[a_idx].position, food.position)
				else:
					f_row, f_col = food.position
				food_lvl = food.level
				idx = self._foods_pos.index(food.position)
				food_obs[3 * idx] = f_row
				food_obs[3 * idx + 1] = f_col
				food_obs[3 * idx + 2] = 0
				food_obs[3 * idx + food_lvl + 2] = 1

			for idx in range(n_seen_agents):
				if self._agent_center:
					a_row, a_col = self.get_centered_pos(self._players[a_idx].position, seen_agents[idx].position)
				else:
					a_row, a_col = seen_agents[idx].position
				agent_lvl = seen_agents[idx].level
				agent_obs[3 * idx] = a_row
				agent_obs[3 * idx + 1] = a_col
				agent_obs[3 * idx + 2] = 0
				agent_obs[3 * idx + agent_lvl + 2] = 1

			obs += [food_obs + agent_obs]

			for p in p_obs.players:
				if p.is_self:
					rewards[a_idx] = p.reward
					break

		return np.array(obs), rewards, done, force_stop, info

	def get_players_adj_food(self) -> List:

		players_adj = []
		for player in self.players:
			adj_pos = self.get_adj_pos(*player.position)
			for pos in adj_pos:
				if pos in self._foods_pos:
					players_adj += [self.players.index(player)]
					break

		return players_adj

	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[np.ndarray, dict[str, Any]]:
		if seed is not None:
			self.seed(seed)

		self.field = np.zeros(self.field_size, np.int32)
		player_lvls = [self.np_random.integers(1, self._max_player_level, endpoint=True) for _ in range(self.n_players)]
		max_food_lvl = self._max_food_lvl if self._max_food_lvl > 0 else sum(player_lvls)
		self.spawn_food(self._n_food_spawn, max_food_lvl)
		self.spawn_players(player_lvls)
		self._current_step = 0
		self._game_over = False
		
		obs, _, _, _, info = self.make_gym_obs()
		return obs, info

	def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:

		n_foods_left = self.count_foods()

		if n_foods_left > 0:
			obs, rewards, dones, force_stop, infos = super().step(actions)
			new_n_foods_left = self.count_foods()
			p_idx_adj_food = self.get_players_adj_food()
			if new_n_foods_left < n_foods_left:
				rewards = np.array([REWARD_PICK if x > 0 else x for x in rewards.tolist()])
			elif len(p_idx_adj_food) > 0:
				for p_idx in p_idx_adj_food:
					rewards[p_idx] = (REWARD_PICK / self._max_episode_steps) * len(p_idx_adj_food) / self._n_agents
			return obs, rewards, dones, force_stop, infos
		else:
			return self.make_obs_dqn_array() if self._use_encoding and not self._grid_observation else self.make_gym_obs()

	#################
	### Utilities ###
	#################
	@property
	def food_pos(self) -> List[Tuple[int, int]]:
		return self._foods_pos

	@property
	def n_food_spawn(self) -> int:
		return self._n_food_spawn

	@property
	def food_spawn_pos(self) -> List[Tuple[int, int]]:
		return self._pos_food_spawn

	@food_pos.setter
	def food_pos(self, new_pos: List[Tuple[int, int]]):
		self._foods_pos = new_pos.copy()

	@n_food_spawn.setter
	def n_food_spawn(self, new_spawn_max: int) -> None:
		self._n_food_spawn = new_spawn_max

	@food_spawn_pos.setter
	def food_spawn_pos(self, new_pos: List[Tuple[int, int]] = None) -> None:
		if new_pos is not None:
			self._pos_food_spawn = new_pos.copy()
		else:
			self._pos_food_spawn = None


class FoodCOOPLBForaging(LimitedCOOPLBForaging):

	Observation = namedtuple("Observation", ["field", "foods", "players", "game_over", "sight", "current_step", "objective"])
	_obj_food_pos: Tuple

	def __init__(self, players: int, max_player_level: int, field_size: Tuple[int, int], max_food: int, sight: int, max_episode_steps: int,
	             force_coop: bool, food_level: int, rng_seed: int, foods_pos: List = None, objective: Tuple = None, render_mode: List = None,
	             use_encoding: bool = False, agent_center: bool = False, grid_observation: bool = False, use_render: bool = False):

		super().__init__(players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop, food_level, rng_seed, foods_pos,
		                 render_mode, use_encoding, agent_center=agent_center, grid_observation=grid_observation, use_render=use_render)

		if objective is None:
			self._obj_food_pos = self._foods_pos[self.np_random.choice(len(self._foods_pos))]
		else:
			self._obj_food_pos = objective

	def _get_observation_space(self) -> Space:
		"""
		The Observation Space for each agent.
		- the board (board_size^2) with foods
		- player description (x, y, level)*player_count
		"""
		if not self._grid_observation:
			field_x = self._field.shape[1]
			field_y = self._field.shape[0]

			max_food = self._max_spawn_food
			max_food_level = self._max_player_level * len(self._players)

			if self._use_encoding:
				min_obs = [-1, -1, 0, *[0] * max_food_level] * max_food + [-1, -1, 0, *[0] * self._max_player_level] * len(self._players) + [-1, -1]
				max_obs = ([field_x - 1, field_y - 1, 1, *[1] * max_food_level] * max_food +
				           [field_x - 1, field_y - 1, 1, *[1] * self._max_player_level] * len(self._players) + [field_x - 1, field_y - 1])
			else:
				min_obs = [-1, -1, 0] * max_food + [-1, -1, 0] * len(self._players) + [-1, -1]
				max_obs = ([field_x - 1, field_y - 1, max_food_level] * max_food + [field_x - 1, field_y - 1, self._max_player_level] * len(self._players) +
				           [field_x - 1, field_y - 1])

			return gymnasium.spaces.Tuple([Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)] * len(self._players))

		else:
			# grid observation space
			grid_shape = (1 + 2 * self._sight, 1 + 2 * self._sight)

			return MultiBinary([len(self._players), 4, *grid_shape])	# Four overlapped observation layers (players, food, occupied spaces and target food)

	#################
	### Utilities ###
	#################
	@property
	def obj_food(self) -> Tuple:
		return self._obj_food_pos

	@obj_food.setter
	def obj_food(self, objective: Tuple):
		self._obj_food_pos = objective

	def set_objective(self, objective: Tuple):
		self._obj_food_pos = objective

	def get_obj_food(self) -> Optional[Food]:
		for food in self.foods:
			if food.position == self._obj_food_pos:
				return food
		return None

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

		log += 'Food target: (%d, %d)\n' % (self.obj_food[0], self.obj_food[1])
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
		log += 'Food target: (%d, %d)\n' % (self.obj_food[0], self.obj_food[1])
		log += 'Current timestep: %d\nGame is finished: %r\nGame has timed out: %r\n' % (self._current_step, self.game_over,
		                                                                                 self._current_step > self._max_episode_steps)

		return log

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
				objective=self._obj_food_pos
		)

	def make_obs_array(self) -> np.ndarray:
		if self._grid_observation:
			self._grid_observation = False
			obs = np.array([np.zeros(self._get_observation_space()[0].shape, dtype=np.int32)] * self._n_agents)
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

			if self._agent_center:
				t_pos = self.get_centered_pos(self._players[idx].position, self._obj_food_pos)
				if t_pos is not None:
					obs[idx][-2] = t_pos[0]
					obs[idx][-1] = t_pos[1]
				else:
					obs[idx][-2] = -1
					obs[idx][-1] = -1
			else:
				obs[idx][-2] = self._obj_food_pos[0]
				obs[idx][-1] = self._obj_food_pos[1]

		return obs

	def make_grid_observations(self) -> np.ndarray:
		layers_size = (self._field_size[0] + 2 * self._sight, self._field_size[1] + 2 * self._sight)

		# Initialize layers
		agent_layer = np.zeros(layers_size)
		food_layer = np.zeros(layers_size)
		occupancy_layer = np.ones(layers_size)
		target_layer = np.zeros(layers_size)
		occupancy_layer[:self._sight, :] = 0
		occupancy_layer[-self._sight:, :] = 0
		occupancy_layer[:, :self._sight] = 0
		occupancy_layer[:, -self._sight:] = 0

		# Update target layer
		obj_food = self.get_obj_food()
		if not obj_food.picked:
			target_layer[self._sight + self._obj_food_pos[0], self._sight + self._obj_food_pos[1]] = 1

		# Update agent and occupancy layers
		for a in self._players:
			pos = a.position
			agent_layer[pos[0] + self._sight, pos[1] + self._sight] = 1
			occupancy_layer[pos[0] + self._sight, pos[1] + self._sight] = 0

		# Update food and occupancy layers
		for f in self._foods:
			if not f.picked:
				pos = f.position
				food_layer[pos[0] + self._sight, pos[1] + self._sight] = f.level if not self._force_coop else 1
				occupancy_layer[pos[0] + self._sight, pos[1] + self._sight] = 0

		obs = np.stack([agent_layer, food_layer, occupancy_layer, target_layer])
		padding = 2 * self._sight + 1

		return np.array([obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding] for a in self._players])

	def make_target_grid_observations(self, obj_pos: Tuple) -> np.ndarray:
		layers_size = (self._field_size[0] + 2 * self._sight, self._field_size[1] + 2 * self._sight)

		# Initialize layers
		agent_layer = np.zeros(layers_size)
		food_layer = np.zeros(layers_size)
		occupancy_layer = np.ones(layers_size)
		target_layer = np.zeros(layers_size)
		occupancy_layer[:self._sight, :] = 0
		occupancy_layer[-self._sight:, :] = 0
		occupancy_layer[:, :self._sight] = 0
		occupancy_layer[:, -self._sight:] = 0

		# Update target layer
		target_layer[self._sight + obj_pos[0], self._sight + obj_pos[1]] = 1

		# Update agent and occupancy layers
		for a in self._players:
			pos = a.position
			agent_layer[pos[0] + self._sight, pos[1] + self._sight] = 1
			occupancy_layer[pos[0] + self._sight, pos[1] + self._sight] = 0

		# Update food and occupancy layers
		for f in self._foods:
			if not f.picked:
				pos = f.position
				food_layer[pos[0] + self._sight, pos[1] + self._sight] = f.level if not self._force_coop else 1
				occupancy_layer[pos[0] + self._sight, pos[1] + self._sight] = 0

		obs = np.stack([agent_layer, food_layer, occupancy_layer, target_layer])
		padding = 2 * self._sight + 1

		return np.array([obs[:, a.position[0]:a.position[0] + padding, a.position[1]:a.position[1] + padding] for a in self._players])

	def make_obs_dqn_array(self) -> tuple[np.ndarray, np.ndarray, bool, bool, Dict]:

		obs = []
		rewards = np.zeros(self._n_agents)
		done = self.game_over
		force_stop = self._current_step > self._max_episode_steps
		raw_obs = [self.make_obs(p) for p in self._players]
		info = {}

		for a_idx in range(self._n_agents):
			food_obs = ([-(self._sight + 1), -(self._sight + 1), 1] + [0] * self._max_food_lvl) * self._max_spawn_food
			agent_obs = ([-(self._sight + 1), -(self._sight + 1), 1] + [0] * self._max_player_level) * self._n_agents
			p_obs = raw_obs[a_idx]
			seen_agents = tuple([p for p in p_obs.players if p.is_self] + [p for p in p_obs.players if not p.is_self])
			seen_foods = tuple(p_obs.foods)
			n_seen_agents = len(seen_agents)

			for food in seen_foods:
				if self._agent_center:
					f_row, f_col = self.get_centered_pos(self._players[a_idx].position, food.position)
				else:
					f_row, f_col = food.position
				food_lvl = food.level
				if food == self._obj_food_pos:
					idx = 0
				else:
					obj_idx = self.food_pos.index(self._obj_food_pos)
					food_idx = self._foods_pos.index(food.position)
					if food_idx < obj_idx:
						idx = food_idx + 1
					else:
						idx = food_idx
				food_obs[3 * idx] = f_row
				food_obs[3 * idx + 1] = f_col
				food_obs[3 * idx + 2] = 0
				food_obs[3 * idx + food_lvl + 2] = 1

			for idx in range(n_seen_agents):
				if self._agent_center:
					a_row, a_col = self.get_centered_pos(self._players[a_idx].position, seen_agents[idx].position)
				else:
					a_row, a_col = seen_agents[idx].position
				agent_lvl = seen_agents[idx].level
				agent_obs[3 * idx] = a_row
				agent_obs[3 * idx + 1] = a_col
				agent_obs[3 * idx + 2] = 0
				agent_obs[3 * idx + agent_lvl + 2] = 1

			if self._agent_center:
				t_pos = self.get_centered_pos(self._players[a_idx].position, self._obj_food_pos)
				if t_pos is not None:
					target_obs = [t_pos[0], t_pos[1]]
				else:
					target_obs = [-(self._sight + 1), -(self._sight + 1)]
			else:
				target_obs = [self._obj_food_pos[0], self._obj_food_pos[1]]

			obs += [food_obs + agent_obs + target_obs]

		for a_idx in range(self._n_agents):
			rewards[a_idx] = self._players[a_idx].reward

		return np.array(obs), rewards, done, force_stop, info

	def get_players_adj_food(self) -> List:

		players_adj = []
		food_adj_pos = self.get_adj_pos(*self._obj_food_pos)

		for player in self.players:
			if player.position in food_adj_pos:
				players_adj += [self.players.index(player)]

		return players_adj

	def spawn_food(self, max_food: int, max_level: int):
		min_level = max_level if self._force_coop else 1
		self.field = np.zeros(self.field_size, np.int32)
		foods_spawned = []
		food_count = 0

		if self._n_food_spawn < 1:
			self._n_food_spawn = max_food

		if max_food < self._max_spawn_food:
			if max_food - 1 > 0:
				# Randomly pick food items to spawn until maximum number
				if self._pos_food_spawn is None:
					idx_spawn = list(range(self._max_spawn_food))
					idx_spawn.remove(self.food_pos.index(self._obj_food_pos))
					pos_spawn = sorted(self.np_random.choice(idx_spawn, size=max_food-1, replace=False))
					self._pos_food_spawn = [self._foods_pos[pos_idx] for pos_idx in pos_spawn]
				else:
					pos_spawn = []
					for pos in self._pos_food_spawn:
						pos_spawn.append(self._foods_pos.index(pos))
			else:
				pos_spawn = []
				if self._pos_food_spawn is None:
					self._pos_food_spawn = []

			foods_spawn = [self._foods_pos.index(self._obj_food_pos)] + pos_spawn
			foods_spawn.sort()
			for idx in foods_spawn:
				pos = self._foods_pos[idx]
				row, col = pos
				if self.field[row, col] == CellEntity.EMPTY:
					food_lvl = min_level if min_level == max_level else self.np_random.integers(min_level, max_level + 1)
					new_food = Food()
					new_food.setup(pos, food_lvl, food_count + 1)
					self.field[row, col] = CellEntity.FOOD
					food_count += 1
					foods_spawned.append(new_food)

		else:
			if self._pos_food_spawn is None:
				self._pos_food_spawn = []
			# Spawn all food items
			for pos in self._foods_pos:
				row, col = pos
				if self.field[row, col] == CellEntity.EMPTY:
					food_lvl = min_level if min_level == max_level else self.np_random.integers(min_level, max_level + 1)
					new_food = Food()
					new_food.setup(pos, food_lvl, food_count + 1)
					self.field[row, col] = CellEntity.FOOD
					food_count += 1
					foods_spawned.append(new_food)

		self._food_spawned = food_count
		self._foods = foods_spawned.copy()

	def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict[str, Any]]:

		n_before_foods = self.count_foods()
		before_field = self.field.copy()

		if n_before_foods > 0:
			rewards = [MOVE_REWARD] * self._n_agents
			obs, raw_rewards, dones, force_stop, infos = super().step(actions)
			after_field = self.field.copy()
			n_after_foods = self.count_foods()
			p_idx_adj_food = self.get_players_adj_food()
			if n_after_foods < n_before_foods:
				field_changes = np.transpose(np.nonzero(before_field - after_field))
				pick_correct = False
				for pos in field_changes:
					if pos[0] == self._obj_food_pos[0] and pos[1] == self._obj_food_pos[1]:
						pick_correct = True
						break
				if pick_correct:
					rewards = [REWARD_PICK if x > 0 else 0 for x in raw_rewards.tolist()]
					dones = True
					self._game_over = True
				else:
					rewards = [WRONG_PICK if x > 0 else 0 for x in raw_rewards.tolist()]
					dones = False

				rewards = np.array(rewards)

			elif len(p_idx_adj_food) > 0:
				for player in self.players:
					p_idx = self.players.index(player)
					if p_idx in p_idx_adj_food:
						rewards[p_idx] = ADJ_FOOD_REWARD * len(p_idx_adj_food) / self._n_agents

			return obs, rewards, dones, force_stop, infos
		else:
			return super().step(actions)