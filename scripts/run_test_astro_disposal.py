#! /usr/bin/env python
import shlex

import subprocess
from pathlib import Path

tests_dir = Path(__file__).parent.absolute().parent.absolute() / 'tests'
USE_SHELL = False

# DQN params
N_LAYERS = 4
BUFFER = 150
GAMMA = 0.95
LAYERS = (40, 128, 128, 84)

# Environment params
# GAME_LEVEL = ['level_one', 'level_two']
GAME_LEVEL = ['level_one']

args = (" --nlayers %d --buffer %d --gamma %f --layer-sizes %s --game-level %s --gpu"
		% (N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]),   														# DQN parameters
		   ' '.join(GAME_LEVEL)))																								# Environment parameters
commamd = "python " + str(tests_dir / 'test_astro_disposal_model.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)

print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
