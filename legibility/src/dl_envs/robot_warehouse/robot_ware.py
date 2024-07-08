import logging
import numpy as np
import gymnasium
import math
import networkx as nx

from collections import namedtuple, defaultdict
from enum import Enum, IntEnum
from itertools import product
from gymnasium import Env, spaces
from gymnasium.envs import Any
from gymnasium.utils import seeding
from gymnasium.spaces import MultiBinary, MultiDiscrete, Space, Box, Discrete
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
from yaml import safe_load
from logging import Logger


class Action(IntEnum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOAD_UNLOAD = 4


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ActionDelta(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class RewardType(IntEnum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class CellEntity(IntEnum):
    FLOOR = 0
    SHELF = 1
    BIN = 2
    FREE_LANE = 3


class Shelf:

    _id: int
    _pos: Tuple[int, int]
    _prev_pos: Optional[Tuple[int, int]]

    def __init__(self, shelf_id: int, pos: Tuple[int, int]):
        self._id = shelf_id
        self._pos = pos
        self._prev_pos = None

    @property
    def pos(self) -> Tuple[int, int]:
        return self._pos

    @property
    def prev_pos(self) -> Tuple[int, int]:
        return self._prev_pos

    @property
    def id(self) -> int:
        return self._id

    @pos.setter
    def pos(self, pos: Tuple[int, int]) -> None:
        self._pos = pos


class Robot:

    _id: int
    _pos: Tuple[int, int]
    _prev_pos: Optional[Tuple[int, int]]
    _direction: Direction
    _message: np.ndarray
    _next_action: Optional[Action]
    _carrying_shelf: Optional[Shelf]
    _has_delivered: bool

    def __init__(self, robot_id: int, pos: Tuple[int, int], direction: Direction, msg_bits: int):

        self._id = robot_id
        self._pos = pos
        self._prev_pos = None
        self._direction = direction
        self._message = np.zeros(msg_bits)
        self._next_action = None
        self._carrying_shelf = None
        self._has_delivered = False

    @property
    def pos(self) -> Tuple[int, int]:
        return self._pos

    @property
    def prev_pos(self) -> Tuple[int, int]:
        return self._prev_pos

    @property
    def direction(self) -> Direction:
        return self._direction

    @property
    def message(self) -> np.ndarray:
        return self._message

    @property
    def next_action(self) -> Action:
        return self._next_action

    @property
    def id(self) -> int:
        return self._id

    @property
    def carrying_shelf(self) -> Shelf:
        return self._carrying_shelf

    @property
    def has_delivered(self) -> bool:
        return self._has_delivered

    @pos.setter
    def pos(self, pos: Tuple[int, int]) -> None:
        self._pos = pos

    @prev_pos.setter
    def prev_pos(self, prev_pos: Tuple[int, int]) -> None:
        self._prev_pos = prev_pos

    @direction.setter
    def direction(self, direction: Direction) -> None:
        self._direction = direction

    @has_delivered.setter
    def has_delivered(self, has_delivered: bool) -> None:
        self._has_delivered = has_delivered

    @next_action.setter
    def next_action(self, action: Action) -> None:
        self._next_action = action

    @message.setter
    def message(self, message: np.ndarray) -> None:
        self._message = message

    @carrying_shelf.setter
    def carrying_shelf(self, shelf: Shelf) -> None:
        self._carrying_shelf = shelf

    def action_location(self, grid_size: Tuple[int, int]) -> Tuple[int, int]:
        if self._next_action != Action.FORWARD:
            return self._pos
        elif self._direction == Direction.UP:
            return max(0, self.pos[0] - 1), self._pos[1]
        elif self.direction == Direction.DOWN:
            return min(grid_size[0], self.pos[0] + 1), self._pos[1]
        elif self.direction == Direction.LEFT:
            return self._pos[0], max(0, self.pos[1] - 1)
        elif self.direction == Direction.RIGHT:
            return self._pos[0], min(grid_size[1], self.pos[1] + 1)
        else:
            raise ValueError(f'Direction {self.direction} is unknown. Should be one of {[v for v in Direction]}')

    def action_direction(self) -> Direction:
        dir_list = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self._next_action == Action.RIGHT:
            return dir_list[(dir_list.index(self._direction) + 1) % len(dir_list)]
        elif self._next_action == Action.LEFT:
            return dir_list[(dir_list.index(self._direction) - 1) % len(dir_list)]
        else:
            return self._direction


class RobotWarehouse(gymnasium.Env):

    action_space: spaces.Tuple
    observation_space: Union[MultiBinary, gymnasium.spaces.Tuple]

    def __init__(self, grid_size: Tuple[int, int], shelves_block: Tuple[int, int], block_length: int, block_spacing: Tuple[int, int], n_robots: int,
                 msg_bits: int, sensor_range: int, request_size: int, max_inactivity_steps: Optional[int], max_stpes: int,
                 reward_type: RewardType, layout: Optional[str], logger: Logger, layered_obs: bool = False):

        assert grid_size[0] > grid_size[1] or grid_size[1] > grid_size[0], "Field can\'t be a square grid."
        self._grid_size = grid_size
        self._logger = logger

        if layout is not None:
            self._setup_env_from_layout(layout)
        else:
            self._setup_env_from_params(shelves_block, block_length, block_spacing)

        self._n_robots = n_robots
        self._msg_bits = msg_bits
        self._sensor_range = sensor_range
        self._max_inactivity_steps = max_inactivity_steps
        self._max_stpes = max_stpes
        self._reward_type = reward_type
        self._layered_obs = layered_obs
        self._request_size = request_size

        self.action_space = spaces.Tuple(n_robots * ([MultiDiscrete([len(Action), msg_bits * (2, )])] if msg_bits > 0 else [Discrete(len(Action))]))
        self.observation_space = self._get_observation_space()

    def _setup_env_from_layout(self, layout: str) -> None:

        config_filepath = Path(__file__).parent.absolute() / 'data' / 'configs' / 'layouts' / (layout + '.yaml')
        with open(config_filepath) as config_file:
            config_data = safe_load(config_file)

        n_rows, n_cols = self._grid_size
        self._field = np.zeros(self._grid_size, dtype=np.int32)
        self._bins = []

        field_data = config_data['field']
        for row in range(n_rows):
            for col in range(n_cols):
                try:
                    cell_val = field_data[row, col].lower()
                    assert cell_val in "bsh "
                    if cell_val == 'b':
                        self._bins.append(tuple([row, col]))
                        self._field = CellEntity.BIN
                    elif cell_val == 'h':
                        self._field = CellEntity.FREE_LANE
                    elif cell_val == 's':
                        self._field = CellEntity.SHELF
                    else:
                        self._field = CellEntity.FLOOR
                except AssertionError:
                    self._logger.error("Field data in position (%d, %d): %s invalid. Defaulting to a normal floor space." % (row, col, cell_val))

        assert len(self._bins) > 0, "Environment must have at least one shelf drop point."

    def _setup_env_from_params(self, shelves_blocks: Tuple[int, int], block_length: int, block_space: Tuple[int, int]) -> None:

        n_rows, n_cols = self._grid_size
        assert min(block_space) >= 1, "Shelf blocks must have at least one row and one column between them."
        assert (((n_rows - (shelves_blocks[0] * block_length + (shelves_blocks[0] - 1) * block_space[0] + 3) >= 0) and
                 (n_cols - (shelves_blocks[1] * 2 + (shelves_blocks[1] - 1) * block_space[1] + 2) >= 0)) if n_rows > n_cols else
                ((n_rows - (shelves_blocks[0] * 2 + (shelves_blocks[0] - 1) * block_space[0] + 2) >= 0) and
                 (n_cols - (shelves_blocks[1] * block_length + (shelves_blocks[1] - 1) * block_space[1] + 3) >= 0))), "Configuration of shelves does not fit in the grid dimensions."

        self._field = np.ones(self._grid_size, dtype=np.int32) * CellEntity.FREE_LANE

        if n_rows > n_cols:

            self._bins = [(n_rows - 1, n_cols // 2 - 1), (n_rows - 1, n_cols // 2)]
            self._field[self._bins[0][0], self._bins[0][1]] = CellEntity.BIN
            self._field[self._bins[1][0], self._bins[1][1]] = CellEntity.BIN

            if (n_rows - (shelves_blocks[0] * block_length + (shelves_blocks[0] - 1) * block_space[0] + 3)) > 0:
                blocks_left_row = shelves_blocks[0]
                for i in range(1, n_rows // 2, block_length + block_space[0]):
                    if blocks_left_row > 0:
                        blocks_left_col = shelves_blocks[1]
                        if (blocks_left_row - 2) >= 0:
                            for j in range(1, n_cols // 2, 2 + block_space[1]):
                                if blocks_left_col > 0:
                                    if (blocks_left_col - 2) >= 0:
                                        self._field[i:i + block_length, j:j + 2] = CellEntity.SHELF
                                        self._field[i:i + block_length, -(j + 1):-(j + 3):-1] = CellEntity.SHELF
                                        self._field[-(i + 2):-(i + block_length + 2):-1, j:j + 2] = CellEntity.SHELF
                                        self._field[-(i + 2):-(i + block_length + 2):-1, -(j + 1):-(j + 3):-1] = CellEntity.SHELF
                                        blocks_left_col -= 2
                                    else:
                                        self._field[i:i + block_length, j:j + 2] = CellEntity.SHELF
                                        self._field[-(i + 2):-(i + block_length + 2):-1, j:j + 2] = CellEntity.SHELF
                                        blocks_left_col -= 1
                            blocks_left_row -= 2
                        else:
                            for j in range(1, n_cols // 2, 2 + block_space[1]):
                                if blocks_left_col > 0:
                                    if (blocks_left_col - 2) >= 0:
                                        self._field[i:i+block_length, j:j+2] = CellEntity.SHELF
                                        self._field[i:i+block_length, -(j+1):-(j+3):-1] = CellEntity.SHELF
                                        blocks_left_col -= 2
                                    else:
                                        self._field[i:i + block_length, j:j+2] = CellEntity.SHELF
                                        blocks_left_col -= 1
                            blocks_left_row -= 1

            else:
                for i in range(1, n_rows - 2, block_length + block_space[0]):
                    blocks_left_col = shelves_blocks[1]
                    for j in range(1, n_cols // 2, 2 + block_space[1]):
                        if blocks_left_col > 0:
                            if (blocks_left_col - 2) >= 0:
                                self._field[i:i + block_length, j:j + 2] = CellEntity.SHELF
                                self._field[i:i + block_length, -(j + 1):-(j + 3):-1] = CellEntity.SHELF
                                blocks_left_col -= 2
                            else:
                                self._field[i:i + block_length, j:j + 2] = CellEntity.SHELF
                                blocks_left_col -= 1

        else:

            self._bins = [(n_rows // 2 - 1, 0), (n_rows // 2, 0)]
            self._field[self._bins[0][0], self._bins[0][1]] = CellEntity.BIN
            self._field[self._bins[1][0], self._bins[1][1]] = CellEntity.BIN

            if (n_cols - (shelves_blocks[1] * block_length + (shelves_blocks[1] - 1) * block_space[1] + 3)) > 0:
                blocks_left_col = shelves_blocks[1]
                for j in range(1, n_cols // 2, block_length + block_space[1]):
                    if blocks_left_col > 0:
                        blocks_left_row = shelves_blocks[0]
                        if (blocks_left_col - 2) >= 0:
                            for i in range(1, n_rows // 2, 2 + block_space[0]):
                                if blocks_left_row > 0:
                                    if (blocks_left_row - 2) >= 0:
                                        self._field[i:i + 2, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                        self._field[-(i + 1):-(i + 3):-1, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                        self._field[i:i + 2, -(j + 1):-(j + block_length + 1)] = CellEntity.SHELF
                                        self._field[-(i + 1):-(i + 3):-1, -(j + 1):-(j + block_length + 1)] = CellEntity.SHELF
                                        blocks_left_row -= 2
                                    else:
                                        self._field[i:i + 2, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                        self._field[i:i + 2, -(j + 1):-(j + block_length + 1)] = CellEntity.SHELF
                                        blocks_left_row -= 1
                            blocks_left_col -= 2
                        else:
                            for i in range(1, n_rows // 2, 2 + block_space[0]):
                                if blocks_left_row > 0:
                                    if (blocks_left_row - 2) >= 0:
                                        self._field[i:i + 2, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                        self._field[-(i + 1):-(i + 3):-1, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                        blocks_left_row -= 2
                                    else:
                                        self._field[i:i + 2, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                        blocks_left_row -= 1
                            blocks_left_col -= 1
            else:
                for j in range(1, n_cols - 1, block_length + block_space[1]):
                    blocks_left_row = shelves_blocks[0]
                    for i in range(1, n_rows // 2, 2 + block_space[0]):
                        if blocks_left_row > 0:
                            if (blocks_left_row - 2) >= 0:
                                self._field[i:i + 2, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                self._field[-(i + 1):-(i + 3):-1, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                blocks_left_row -= 2
                            else:
                                self._field[i:i + 2, (j + 1):(j + block_length + 1)] = CellEntity.SHELF
                                blocks_left_row -= 1

    def _get_observation_space(self) -> Union[MultiBinary, gymnasium.spaces.Tuple]:

        if self._layered_obs:

        else:
            
