#! /usr/bin/env python

import numpy as np
import math

from .agent import Agent, ActionDirection, Action


class RandomPrey(Agent):
	def act(self, env) -> int:
		return self._np_random.integers(len(Action))