#! /usr/bin/env python

import sys

import numpy as np
import argparse
import gym
import lbforaging
import pickle
import yaml
import itertools
import time
import os

from pathlib import Path
from gym.envs.registration import register
from lbforaging.foraging.environment import ForagingEnv
from termcolor import colored
from typing import List, Tuple, Dict


MOVE_REWARD = -0.5
REWARD_PICK = 10
ADJ_FOOD_REWARD = 0.75
WRONG_PICK = -10
RNG_SEED = 27062023


class LimitedCOOPLBForaging(ForagingEnv):
	
	_foods_pos: List[Tuple[int, int]]
	_food_lvl: int
	_n_food_spawn: int = 0
	_pos_food_spawn: List[Tuple[int, int]] = None
	_rng_gen: np.random.Generator
	
	def __init__(self, players: int, max_player_level: int, field_size: Tuple[int, int], max_food: int, sight: int, max_episode_steps: int,
				 force_coop: bool, food_level: int, rng_seed: int, foods_pos: List = None):
		
		super().__init__(players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop)
		
		self._food_lvl = food_level
		self._rng_gen = np.random.default_rng(rng_seed)
		if foods_pos is None:
			self._foods_pos = []
			n_foods = len(self._foods_pos)
			while n_foods < max_food:
				tmp_pos_idx = self._rng_gen.choice(field_size[0] * field_size[1])
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
	
	def spawn_players(self, max_player_level, player_pos=None):
		
		for player in self.players:
			
			attempts = 0
			player.reward = 0
			
			if player_pos is None:
				while attempts < 1000:
					row = self._rng_gen.integers(0, self.rows)
					col = self._rng_gen.integers(0, self.cols)
					if self._is_empty_location(row, col):
						player.setup((row, col), max_player_level, self.field_size, )
						break
					attempts += 1
			else:
				player_idx = self.players.index(player)
				row, col = player_pos[player_idx]
				player.setup((row, col), max_player_level, self.field_size, )
	
	def spawn_food(self, max_food, max_level):
		min_level = max_level if self.force_coop else 1
		self.field = np.zeros(self.field_size, np.int32)
		
		if self._n_food_spawn < 1:
			self._n_food_spawn = max_food
		
		if max_food < self.max_food:
			# Randomly pick food items to spawn
			if self._pos_food_spawn is None:
				if max_food > 1:
					pos_spawn = [self.food_pos[idx] for idx in sorted(self._rng_gen.choice(range(self.max_food), size=max_food, replace=False))]
				else:
					pos_spawn = [self.food_pos[self._rng_gen.choice(range(self.max_food))]]
				self._pos_food_spawn = pos_spawn.copy()
			for pos in self._pos_food_spawn:
				row, col = pos
				if self.field[row, col] == 0:
					self.field[row, col] = (
						self._food_lvl if self._food_lvl > 0 else (min_level if min_level == max_level else self._rng_gen.integers(min_level, max_level))
					)
		
		else:
			if self._pos_food_spawn is None:
				self._pos_food_spawn = []
			# Spawn all food items
			for pos in self._foods_pos:
				row, col = pos
				if self.field[row, col] == 0:
					self.field[row, col] = (
						self._food_lvl if self._food_lvl > 0 else (min_level if min_level == max_level else self._rng_gen.integers(min_level, max_level))
					)
		
		self._food_spawned = self.field.sum()
	
	def convert_obs_dqn(self, raw_obs: np.ndarray) -> np.ndarray:
		
		obs = []
		
		for a_idx in range(self.n_agents):
			food_obs = ([-1, -1, 1] + [0] * self._food_lvl) * self.max_food
			food_offset = self._food_lvl + 3
			agent_obs = []
			a_raw_obs = raw_obs[a_idx]
			for obs_idx in range(0, len(a_raw_obs), 3):
				if obs_idx < self.max_food * 3:
					food_row = int(a_raw_obs[obs_idx])
					food_col = int(a_raw_obs[obs_idx + 1])
					food_lvl = int(a_raw_obs[obs_idx + 2])
					if self.field[food_row, food_col] != 0:
						food_idx = self._foods_pos.index((food_row, food_col))
						food_obs[food_idx*food_offset] = food_row
						food_obs[food_idx * food_offset + 1] = food_col
						food_obs[food_idx * food_offset + 2] = 0
						food_obs[food_idx * food_offset + food_lvl + 2] = 1
				else:
					one_hot = [0] * (self.max_player_level + 1)
					one_hot[int(a_raw_obs[obs_idx + 2])] = 1
					agent_obs += [a_raw_obs[obs_idx], a_raw_obs[obs_idx + 1], *one_hot]
			obs += [food_obs + agent_obs]
		
		return np.array(obs, dtype=np.float32)
	
	def get_adj_pos(self, pos: Tuple) -> Tuple[Tuple[int, int]]:
		row, col = pos
		return ((max(row - 1, 0), col), (min(row + 1, self.field_size[0] - 1), col),
				(row, max(col - 1, 0)), (row, min(col + 1, self.field_size[1] - 1)))
	
	def get_players_adj_food(self) -> List:
	
		players_adj = []
		for player in self.players:
			adj_pos = self.get_adj_pos(player.position)
			for pos in adj_pos:
				if pos in self._foods_pos:
					players_adj += [self.players.index(player)]
					break
			
		return players_adj
	
	def reset(self):
		if self._game_over:
			if self._n_food_spawn > 0:
				self.spawn_food(self.food_spawn, self._food_lvl)
			else:
				print(colored('Number of food to spawn not set, defaulting for environment\'s maximum.', 'yellow'))
				self.spawn_food(self.max_food, self._food_lvl)
			self.spawn_players(self.max_player_level)
			self.current_step = 0
			self._game_over = False
			self._gen_valid_moves()
		else:
			self.current_step = 0
			self._game_over = False
			self._gen_valid_moves()
		
		obs, rewards, dones, infos = self._make_gym_obs()
		return self.convert_obs_dqn(obs), rewards, dones, infos
	
	def step(self, actions):
		
		foods_left = np.count_nonzero(self.field)
		
		if foods_left > 0:
			obs, rewards, dones, infos = super().step(actions)
			new_foods_left = np.count_nonzero(self.field)
			p_idx_adj_food = self.get_players_adj_food()
			if new_foods_left < foods_left:
				rewards = [REWARD_PICK if x > 0 else x for x in rewards]
			elif len(p_idx_adj_food) > 0:
				for p_idx in p_idx_adj_food:
					rewards[p_idx] = ADJ_FOOD_REWARD * len(p_idx_adj_food) / self.n_agents
			return self.convert_obs_dqn(obs), rewards, dones, infos
		else:
			obs, rewards, dones, infos = self._make_gym_obs()
			return self.convert_obs_dqn(obs), rewards, dones, infos

	#################
	### Utilities ###
	#################
	@property
	def food_pos(self) -> List[Tuple[int, int]]:
		return self._foods_pos
	
	@property
	def food_level(self) -> int:
		return self._food_lvl

	@property
	def food_spawn(self) -> int:
		return self._n_food_spawn
	
	@property
	def rng_gen(self) -> np.random.Generator:
		return self._rng_gen
	
	@property
	def food_spawn_pos(self) -> List[Tuple[int, int]]:
		return self._pos_food_spawn
	
	@food_pos.setter
	def food_pos(self, new_pos: List[Tuple[int, int]]):
		self._foods_pos = new_pos.copy()
		
	@food_level.setter
	def food_level(self, new_lvl: int):
		self._food_lvl = new_lvl

	@food_spawn.setter
	def food_spawn(self, new_spawn_max: int) -> None:
		self._n_food_spawn = new_spawn_max
	
	@food_spawn_pos.setter
	def food_spawn_pos(self, new_pos: List[Tuple[int, int]] = None) -> None:
		if new_pos is not None:
			self._pos_food_spawn = new_pos.copy()
		else:
			self._pos_food_spawn = None

class FoodCOOPLBForaging(LimitedCOOPLBForaging):
	
	_obj_food: Tuple
	
	def __init__(self, players: int, max_player_level: int, field_size: Tuple[int, int], max_food: int, sight: int, max_episode_steps: int,
				 force_coop: bool, food_level: int, rng_seed: int, foods_pos: List = None, objective: Tuple = None):
		
		super().__init__(players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop, food_level, rng_seed, foods_pos)
		
		if objective is None:
			self._obj_food = self._foods_pos[self._rng_gen.choice(len(self._foods_pos))]
		else:
			self._obj_food = objective
	
	def get_players_adj_food(self) -> List:
		
		players_adj = []
		
		for player in self.players:
			adj_pos = self.get_adj_pos(player.position)
			for pos in adj_pos:
				if pos[0] == self._obj_food[0] and pos[1] == self._obj_food[1]:
					players_adj += [self.players.index(player)]
					break
		
		return players_adj

	def spawn_food(self, max_food, max_level):
		min_level = max_level if self.force_coop else 1
		self.field = np.zeros(self.field_size, np.int32)

		if self._n_food_spawn < 1:
			self._n_food_spawn = max_food
		
		if max_food < self.max_food:
			# Objective food must be spawned
			row, col = self._obj_food
			self.field[row, col] = (
				self._food_lvl if self._food_lvl > 0 else (min_level if min_level == max_level else self.np_random.randint(min_level, max_level))
			)
			
			if max_food - 1 > 0:
				# Randomly pick food items to spawn until maximum number
				if self._pos_food_spawn is None:
					rng_gen = np.random.default_rng(RNG_SEED)
					idx_spawn = list(range(self.max_food))
					idx_spawn.remove(self.food_pos.index(self._obj_food))
					pos_spawn = [self.food_pos[idx] for idx in sorted(rng_gen.choice(idx_spawn, size=max_food-1, replace=False))]
					self._pos_food_spawn = pos_spawn
				for pos in self._pos_food_spawn:
					row, col = pos
					if self.field[row, col] == 0:
						self.field[row, col] = (
							self._food_lvl if self._food_lvl > 0 else (min_level if min_level == max_level else self.np_random.randint(min_level, max_level))
						)
			elif self._pos_food_spawn is None:
				self._pos_food_spawn = []
			
		else:
			if self._pos_food_spawn is None:
				self._pos_food_spawn = []
			# Spawn all food items
			for pos in self._foods_pos:
				# print(self._foods_pos)
				row, col = pos
				if self.field[row, col] == 0:
					self.field[row, col] = (
						self._food_lvl if self._food_lvl > 0 else (min_level if min_level == max_level else self.np_random.randint(min_level, max_level))
					)
		
		self._food_spawned = self.field.sum()

	def step(self, actions):
		
		before_foods = np.count_nonzero(self.field)
		before_field = self.field.copy()
		
		if before_foods > 0:
			rewards = [MOVE_REWARD] * self.n_agents
			obs, raw_rewards, dones, infos = super().step(actions)
			after_field = self.field.copy()
			after_foods = np.count_nonzero(self.field)
			p_idx_adj_food = self.get_players_adj_food()
			if after_foods < before_foods:
				pick_food = tuple(np.transpose(np.nonzero(before_field - after_field))[0])
				if pick_food != self._obj_food:
					rewards = [WRONG_PICK if x > 0 else 0 for x in raw_rewards]
					dones = [False for _ in dones]
					
				else:
					rewards = [REWARD_PICK if x > 0 else 0 for x in raw_rewards]
					dones = [True for _ in dones]
					self._game_over = True
					
			elif len(p_idx_adj_food) > 0:
				for player in self.players:
					p_idx = self.players.index(player)
					if p_idx in p_idx_adj_food:
						rewards[p_idx] += ADJ_FOOD_REWARD * len(p_idx_adj_food) / self.n_agents
					else:
						rewards[p_idx] += 0
			
			return obs, rewards, dones, infos
		else:
			return super().step(actions)
		
	#################
	### Utilities ###
	#################
	@property
	def obj_food(self) -> Tuple:
		return self._obj_food
	
	@obj_food.setter
	def obj_food(self, objective: Tuple):
		self._obj_food = objective
	
