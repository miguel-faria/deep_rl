#! /usr/bin/env python

import pygame
import numpy as np

from astro_waste_env import PlayerState, ObjectState, AstroWasteEnv, CellEntity
from typing import Tuple
from pathlib import Path


class Viewer(object):
	
	def __init__(self, window_size: Tuple, env: AstroWasteEnv, icon_size: int = -1, visible: bool = True):
	
		self._width, self._height = window_size
		self._env = env
		self._icon_size = icon_size
		self._visible = visible