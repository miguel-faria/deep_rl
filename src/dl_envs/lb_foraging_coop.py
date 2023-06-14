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


class LimitedCOOPLBForaging(ForagingEnv):
	
	_foods_pos: List[Tuple[int, int]]
	_food_lvl: int
	
	def __init__(self, players: int, max_player_level: int, field_size: Tuple[int, int], max_food: int, sight: int, max_episode_steps: int,
				 force_coop: bool, food_level: int, foods_pos: List = None):
		
		super().__init__(players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop)
		
		self._food_lvl = food_level
		if foods_pos is None:
			self._foods_pos = []
			n_foods = len(self._foods_pos)
			while n_foods < max_food:
				tmp_pos_idx = np.random.choice(field_size[0] * field_size[1])
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
	
	def spawn_players(self, max_player_level):
		
		for player in self.players:
			
			attempts = 0
			player.reward = 0
			
			while attempts < 1000:
				row = self.np_random.randint(0, self.rows)
				col = self.np_random.randint(0, self.cols)
				if self._is_empty_location(row, col):
					player.setup((row, col), max_player_level, self.field_size, )
					break
				attempts += 1
	
	def spawn_food(self, max_food, max_level):
		min_level = max_level if self.force_coop else 1
		
		for pos in self._foods_pos:
			row, col = pos
			self.field[row, col] = (
				self._food_lvl if self._food_lvl > 0 else (min_level if min_level == max_level else self.np_random.randint(min_level, max_level))
			)
		
		self._food_spawned = self.field.sum()
	
	def reset(self):
		self.field = np.zeros(self.field_size, np.int32)
		self.spawn_food(self.max_food, self.max_player_level)
		self.spawn_players(self.max_player_level)
		self.current_step = 0
		self._game_over = False
		self._gen_valid_moves()
		
		return self._make_gym_obs()
	
	def step(self, actions):
		
		foods_left = np.count_nonzero(self.field)
		
		if foods_left > 0:
			obs, rewards, dones, infos = super().step(actions)
			new_foods_left = np.count_nonzero(self.field)
			if new_foods_left < foods_left:
				rewards = [int(x > 0) for x in rewards]
				
			return obs, rewards, dones, infos
		else:
			return self._make_gym_obs()

	#################
	### Utilities ###
	#################
	@property
	def food_pos(self) -> List[Tuple[int, int]]:
		return self._foods_pos
	
	@property
	def food_level(self) -> int:
		return self._food_lvl
	
	@food_pos.setter
	def food_pos(self, new_pos: List[Tuple[int, int]]):
		self._foods_pos = new_pos.copy()
		
	@food_level.setter
	def food_level(self, new_lvl: int):
		self._food_lvl = new_lvl

	
class FoodCOOPLBForaging(LimitedCOOPLBForaging):
	
	_obj_food: Tuple
	
	def __init__(self, players: int, max_player_level: int, field_size: Tuple[int, int], max_food: int, sight: int, max_episode_steps: int,
				 force_coop: bool, food_level: int, foods_pos: List = None, objective: Tuple = None):
		
		super().__init__(players, max_player_level, field_size, max_food, sight, max_episode_steps, force_coop, food_level, foods_pos)
		
		if objective is None:
			self._obj_food = self._foods_pos[np.random.choice(len(self._foods_pos))]
		else:
			self._obj_food = objective

	def step(self, actions):
		
		before_foods = np.count_nonzero(self.field)
		before_field = self.field.copy()
		print(self.field)
		print()
		
		if before_foods > 0:
			obs, rewards, dones, infos = super().step(actions)
			after_field = self.field.copy()
			after_foods = np.count_nonzero(self.field)
			if after_foods < before_foods:
				pick_food = tuple(np.transpose(np.nonzero(before_field - after_field))[0])
				if pick_food != self._obj_food:
					rewards = [x * 0 for x in rewards]
					dones = [False for _ in dones]
					
				else:
					rewards = [1 if x > 0 else 0 for x in rewards]
					dones = [True for _ in dones]
			
			return obs, rewards, dones, infos
		else:
			return self._make_gym_obs()
		
	#################
	### Utilities ###
	#################
	@property
	def obj_food(self) -> Tuple:
		return self._obj_food
	
	@obj_food.setter
	def obj_food(self, objective: Tuple):
		self._obj_food = objective
	
