#! /usr/bin/env python
import shlex
import subprocess
import time
import argparse

from pathlib import Path


tests_dir = Path(__file__).parent.absolute().parent.absolute() / 'tests'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'lb_foraging'
data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
USE_SHELL = False

# Test parameters
TEST_MODE = 2
N_TESTS = 5
USE_GPU = True
RUN_PARALELL = False
USE_RENDER = False

# Environemnt parameters
N_AGENTS = 2
PLAYER_LVL = 1
FIELD_LENGTH = 8
MAX_FOODS = 8
MAX_SPAWN_FOODS = 6
FOOD_LEVEL = 2
STEPS_EPISODE = 600

# Models parameters
N_LEG_AGENTS = 1
ARCHITECTURE = 'v3'
GAMMA = 0.95
USE_CNN = True
USE_DUELING = True
USE_DDQN = True
USE_VDN = True

args = ('--mode %d --runs %d --models-dir %s --data-dir %s --logs-dir %s --n-agents %d --player-level %d --field-size %d '
        '--n-food %d --food-level %d --steps-episode %d --n-foods-spawn %d --n-leg-agents %d --architecture %s --gamma %f '
        % (TEST_MODE, N_TESTS, models_dir, data_dir, log_dir, N_AGENTS, PLAYER_LVL, FIELD_LENGTH, MAX_FOODS, FOOD_LEVEL,
           STEPS_EPISODE, MAX_SPAWN_FOODS, N_LEG_AGENTS, ARCHITECTURE, GAMMA))
args += ((' --render' if USE_RENDER else '') + (' --paralell' if RUN_PARALELL else '') + (' --use_gpu' if USE_GPU else '') +
         (' --cnn' if USE_CNN else '') + (' --dueling' if USE_DUELING else '') + (' --ddqn' if USE_DDQN else '') + (' --vdn' if USE_VDN else ''))

command = "python " + str(tests_dir / 'test_legible_collaboration.py') + args
if not USE_SHELL:
	command = shlex.split(command)

print(command)
start_time = time.time()
try:
	subprocess.run(command, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)

except KeyboardInterrupt as ki:
	print('Caught keyboard interrupt by user: %s Exiting....' % ki)

except Exception as e:
	print('Caught general exception: %s' % e)

wall_time = time.time() - start_time
print('Finished training, took %.3f seconds' % wall_time)
