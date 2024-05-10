#! /usr/bin/env python

import numpy as np
import math

from .agent import Agent, ActionDirection, Action


class GreedyPrey(Agent):
	
	def act(self, env) -> int:
	
		agents_dist = [math.sqrt((self._pos[0] - env.agents[h_id].pos[0]) ** 2 + (self._pos[1] - env.agents[h_id].pos[1]) ** 2)  for h_id in env.hunter_ids]
		closest_agent = env.agents[env.hunter_ids[np.argmin(agents_dist)]]
		closest_dist = agents_dist[np.argmin(agents_dist)]
		rows, cols = env.field_size
		best_actions = []
		for direction in ActionDirection:
			nxt_pos = (max(min(self._pos[0] + direction.value[0], rows), 0), max(min(self._pos[1] + direction.value[1], rows), 0))
			if math.sqrt((nxt_pos[0] - closest_agent.pos[0]) ** 2 + (nxt_pos[1] - closest_agent.pos[1]) ** 2) > closest_dist:
				best_actions += [Action[ActionDirection(direction).name].value]
		
		if len(best_actions) > 0:
			return np.random.choice(best_actions)
		else:
			return Action.STAY.value