from typing import Dict, List, Tuple, Any

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box
from termcolor import colored

from dl_envs.pursuit.agents.agent import AgentType
from dl_envs.pursuit.pursuit_env import PursuitEnv


class TeamPursuitEnv(PursuitEnv):
	
	_teams_comp: Dict[str, List[str]]
	_initial_teams: Dict[str, List[str]]
	_team_objective: Dict[str, str]
	_initial_objectives: Dict[str, str]
	
	def __init__(self, hunters: List[Tuple[str, int]], preys: List[Tuple[str, int]], field_size: Tuple[int, int], hunter_sight: int, teams: Dict[str, List[str]],
				 n_catch: int = 4, max_steps: int = 250, use_encoding: bool = False, dead_preys: List[bool] = None, target_preys: List[str] = None,
				 use_layer_obs: bool = False, agent_centered: bool = False):
		
		super().__init__(hunters, preys, field_size, hunter_sight, n_catch, max_steps, use_encoding, dead_preys, use_layer_obs, agent_centered)
		
		self._teams_comp = dict()
		self._team_objective = dict()
		self._initial_teams = dict()
		self._initial_objectives = dict()
		for key in teams.keys():
			self._teams_comp[key] = teams[key].copy()
			self._initial_teams[key] = teams[key].copy()
			self._team_objective[key] = ''
			self._initial_objectives[key] = '-1'
			for a_id in teams[key]:
				self.agents[a_id].team = key
	
	@property
	def teams(self) -> Dict[str, List[str]]:
		return self._teams_comp
	
	@property
	def team_objectives(self) -> Dict[str, str]:
		return self._team_objective
	
	def _get_observation_space(self) -> Space:
		
		if self._use_layer_obs:
			# grid observation space
			grid_shape = (1 + 2 * self._hunter_sight, 1 + 2 * self._hunter_sight)
			
			# hunters layer: hunter positions
			hunters_min = np.zeros(grid_shape, dtype=np.int32)
			hunters_max = np.ones(grid_shape, dtype=np.int32)
			
			# preys layer: prey positions
			preys_min = np.zeros(grid_shape, dtype=np.int32)
			preys_max = np.ones(grid_shape, dtype=np.int32)
			
			# occupancy layer: i the cell available
			access_min = np.zeros(grid_shape, dtype=np.int32)
			access_max = np.ones(grid_shape, dtype=np.int32)
			
			# team layer: position of team members
			team_min = np.zeros(grid_shape, dtype=np.int32)
			team_max = np.ones(grid_shape, dtype=np.int32)
			
			# objective layer
			objective_min = np.zeros(grid_shape, dtype=np.int32)
			objective_max = np.ones(grid_shape, dtype=np.int32)
			
			# total layer
			min_obs = np.stack([hunters_min, preys_min, access_min, team_min, objective_min])
			max_obs = np.stack([hunters_max, preys_max, access_max, team_max, objective_max])
		
		else:
			n_teams = len(self._teams_comp.keys())
			if self._use_encoding:
				min_obs = [[-1, -1, 0, 0] * self._n_preys_alive + ([-1, -1, 0, 0] + [0] * (n_teams + 1) + [0] * (self.n_preys + 1)) * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 1, 1] * self._n_preys_alive +
						   ([self._field_size[0], self._field_size[1], 1, 1] + [1] * (n_teams + 1) + [1] * (self.n_preys + 1)) * self._n_hunters] * self._n_hunters
			else:
				min_obs = [[-1, -1, 2] * self._n_preys_alive + [-1, -1, 1, 0, 0] * self._n_hunters] * self._n_hunters
				max_obs = [[self._field_size[0], self._field_size[1], 2] * self._n_preys_alive +
						   [self._field_size[0], self._field_size[1], 1, n_teams, self.n_preys] * self._n_hunters] * self._n_hunters
			
		return Box(np.array(min_obs), np.array(max_obs), dtype=np.int32)
	
	def aggregate_obs(self, hunter_obs: List, prey_obs: List) -> List:
		final_obs = []
		for team in self._teams_comp.keys():
			members_idx = [self._hunter_ids.index(m_id) for m_id in self._teams_comp[team]]
			ordered_hunter_obs = []
			for idx in members_idx:
				ordered_hunter_obs += hunter_obs[idx]
			final_obs += [ordered_hunter_obs + prey_obs]
		return final_obs
	
	def make_array_obs(self) -> np.ndarray:
		hunter_obs = []
		prey_obs = []
		
		for a_id in self._agents.keys():
			agent_pos = list(self._agents[a_id].pos)
			if self._agents[a_id].agent_type == AgentType.HUNTER:
				all_preys = [agent.agent_id for agent in self._agents.values() if agent.agent_type == AgentType.PREY]
				agent_team = list(self._teams_comp.keys()).index(self._agents[a_id].team)
				team_objective = all_preys.index(self._team_objective[self._agents[a_id].team])
				hunter_obs += [agent_pos + [AgentType.HUNTER] + [agent_team]+ [team_objective]]
			else:
				prey_obs += agent_pos + [AgentType.PREY]
		
		return np.array(self.aggregate_obs(hunter_obs, prey_obs))
	
	def make_dqn_obs(self) -> np.ndarray:
		hunter_obs = []
		prey_obs = []
		
		for agent_id in self._agents.keys():
			agent_pos = list(self._agents[agent_id].pos)
			if self._agents[agent_id].agent_type == AgentType.HUNTER:
				all_preys = [agent.agent_id for agent in self._agents.values() if agent.agent_type == AgentType.PREY]
				agent_team = [0] * len(self._teams_comp.keys())
				team_objective = [0] * len(all_preys)
				agent_team[list(self._teams_comp.keys()).index(self._agents[agent_id].team)] = 1
				team_objective[all_preys.index(self._team_objective[self._agents[agent_id].team])] = 1
				type_one_hot = [1, 0]
				hunter_obs += [agent_pos + type_one_hot + agent_team + team_objective]
			else:
				type_one_hot = [0, 1]
				prey_obs += agent_pos + type_one_hot
		
		return np.array(self.aggregate_obs(hunter_obs, prey_obs))
	
	def make_grid_obs(self) -> np.ndarray:
		layer_size = (self._field_size[0] + 2 * self._hunter_sight + 1, self._field_size[1] + 2 * self._hunter_sight + 1)
		hunter_layer = np.zeros(layer_size)
		prey_layer = np.zeros(layer_size)
		free_layer = np.ones(layer_size)
		team_layers = dict([[key, np.zeros(layer_size)] for key in self._teams_comp.keys()])
		objective_layers = dict([[key, np.zeros(layer_size)] for key in self._teams_comp.keys()])
		
		free_layer[:self._hunter_sight, :] = 0
		free_layer[-self._hunter_sight:, :] = 0
		free_layer[:, self._hunter_sight] = 0
		free_layer[:, -self._hunter_sight:] = 0
		
		for agent in self._agents.values():
			if agent.alive:
				if agent.agent_type == AgentType.HUNTER:
					hunter_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
					team_layers[agent.team][agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				else:
					prey_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
					for key in self._team_objective.keys():
						if agent.agent_id == self._team_objective[key]:
							objective_layers[key][agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 1
				free_layer[agent.pos[0] + self._hunter_sight, agent.pos[1] + self._hunter_sight] = 0
		
		hunter_obs = np.stack([hunter_layer, prey_layer, free_layer])
		# prey_obs = np.stack([prey_layer, hunter_layer, free_layer])
		padding = 2 * self._hunter_sight + 1
		
		return np.array([np.vstack([hunter_obs[:, self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding],
						 [team_layers[self.agents[hunter_id].team][self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding]],
						 [objective_layers[self.agents[hunter_id].team][self._agents[hunter_id].pos[0]:self._agents[hunter_id].pos[0] + padding,
						 self._agents[hunter_id].pos[1]:self._agents[hunter_id].pos[1] + padding]]]) for hunter_id in self._hunter_ids])
	
	def setup_objectives(self, objectives: Dict[str, str]) -> None:
		
		team_ids = list(self._teams_comp.keys())
		for key in objectives.keys():
			if key in team_ids:
				self._team_objective[key] = objectives[key]
				self._initial_objectives[key] = objectives[key]
			else:
				print(colored('Team id %s is not a valid id.' % key, 'yellow'))
			
	def update_teams(self, new_teams: Dict[str, List[str]], remove_teams: Dict[str, List[str]] = None) -> None:
	
		existing_teams = list(self._teams_comp.keys())
		for key in new_teams.keys():
			self._teams_comp[key] = new_teams[key].copy()
			for a_id in self._teams_comp[key]:
				self.agents[a_id].team = key
		
		if remove_teams is not None:
			for key in remove_teams.keys():
				if key in existing_teams:
					for agent in remove_teams[key]:
						self._teams_comp[key].remove(agent)
				else:
					print(colored('Team id %s is not a valid id.' % key, 'yellow'))
	
	def update_objectives(self, new_objectives: Dict[str, str]) -> None:
		for key in new_objectives.keys():
			self._team_objective[key] = new_objectives[key]
	
	def correct_prey(self, agent_id: str, prey_id: str):
	
		for team_id in self._teams_comp.keys():
			if agent_id in self._teams_comp[team_id]:
				return prey_id == self._team_objective[team_id]

		return False
	
	def get_rewards(self, field: np.ndarray) -> np.ndarray:
		
		# If game ends and there are preys to be caught
		if self._env_timestep > self._env_timestep and self._n_preys_alive > 0:
			rewards = [-EVADE_REWARD] * self._n_hunters
			for agent in self._agents:
				if self._agents[agent].agent_type == AgentType.PREY:
					rewards += [EVADE_REWARD]
			return np.array(rewards)
		
		# When all preys are caught, hunters get max reward and preys get max penalty
		if self._n_preys_alive == 0:
			rewards = [CATCH_ALL_REWARD] * self._n_hunters
			for agent in self._agents:
				if self._agents[agent].agent_type == AgentType.PREY:
					rewards += [-CATCH_ALL_REWARD]
			return np.array(rewards)
		
		# By default, hunters get a zero or negative reward for moving
		rewards_hunters = [MOVE_REWARD] * self._n_hunters
		rewards_preys = []
		all_hunter_ids = [agent_id for agent_id in self._agents.keys() if self._agents[agent_id].agent_type == AgentType.HUNTER]
		all_prey_ids = [agent_id for agent_id in self._agents.keys() if self._agents[agent_id].agent_type == AgentType.PREY]
		for prey in all_prey_ids:
			if prey in self._prey_alive_ids:
				prey_pos = self._agents[prey].pos
				prey_adj = self.adj_pos(prey_pos)
				n_surround_hunters = sum([self.correct_prey(hunter_id, prey) and self.agents[hunter_id].pos in prey_adj for hunter_id in self._hunter_ids])
				is_surrounded = n_surround_hunters >= self._n_need_catch
				# If prey is surrounded, gets a caught penalty and all hunters boxing it get a catching reward
				if is_surrounded:
					if self._n_preys_alive == 1:
						for agent_id in all_hunter_ids:
							rewards_hunters[all_hunter_ids.index(agent_id)] = CATCH_ALL_REWARD
						rewards_preys += [-CATCH_ALL_REWARD]
					else:
						for agent_id in all_hunter_ids:
							if self._agents[agent_id].pos in prey_adj and self.correct_prey(agent_id, prey):
								rewards_hunters[all_hunter_ids.index(agent_id)] = CATCH_REWARD
						rewards_preys += [-CATCH_REWARD]
				# If prey is not surrounded, it gets a reward for being free and agents around it get a small reward
				else:
					rewards_preys += [-MOVE_REWARD]
					for agent_id in all_hunter_ids:
						if self._agents[agent_id].pos in prey_adj and self.correct_prey(agent_id, prey):
							rewards_hunters[all_hunter_ids.index(agent_id)] = n_surround_hunters * (CATCH_REWARD / self._n_need_catch)
			# If prey already caught, doesn't get reward
			else:
				rewards_preys += [0]
		
		return np.array(rewards_hunters + rewards_preys)
		
	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
		_, info = super().reset(seed=seed, options=options)
		for key in self._initial_teams.keys():
			self._teams_comp[key] = self._initial_teams[key].copy()
			if self._team_objective[key] == -1:
				self._team_objective[key] = self._initial_objectives[key]
			for a_id in self._teams_comp[key]:
				self.agents[a_id].team = key
		
		obs = self.make_obs()
		return obs, info
