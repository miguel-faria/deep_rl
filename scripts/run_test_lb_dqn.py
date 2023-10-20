#! /usr/bin/env python
import shlex

import subprocess
from pathlib import Path


tests_dir = Path(__file__).parent.absolute().parent.absolute() / 'tests'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'lb_foraging'
USE_SHELL = False

# DQN params
N_AGENTS = 2
N_LAYERS = 2
BUFFER = 1000
GAMMA = 0.9
LAYERS = (256, 256)
AGENTS = ("agent_1", "agent_2")
USE_DUELING = True
USE_DDQN = True

# Environment params
N_FOODS = 8
FIELD_LENGTH = 8
STEPS_EPISODE = 400
PLAYER_LEVEL = 1
FOOD_LVL = 2
USE_RENDER = True
USE_GPU = True

# Testing params
N_CYCLES = 1
MODEL_INFO = ["20230803-113919", "all_final"]
N_SPAWN_FOODS = 8

args = (" --nagents %d --nlayers %d --buffer %d --gamma %f --layer-sizes %s --agent-ids %s "
		"--player-level %d --field-size %d --n-food %d --food-level %d --steps-episode %d "
		"--cycles %d --model-info %s %s --n-foods-spawn %d"
		% (N_AGENTS, N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]), ' '.join([str(x) for x in AGENTS]),				# DQN parameters
		   PLAYER_LEVEL, FIELD_LENGTH, N_FOODS, FOOD_LVL, STEPS_EPISODE,															# Environment parameters
		   N_CYCLES, MODEL_INFO[0], MODEL_INFO[1], N_SPAWN_FOODS))																	# Testing parameters
args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else ""))
commamd = "python " + str(tests_dir / 'test_lb_dqn_model.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)
	
print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
