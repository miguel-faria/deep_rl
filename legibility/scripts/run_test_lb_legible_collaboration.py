#! /usr/bin/env python
import shlex
import subprocess
import time
import argparse

from pathlib import Path


tests_dir = Path(__file__).parent.absolute().parent.absolute() / 'tests'
USE_SHELL = False

# Test parameters
TEST_MODE = 2
N_TESTS = 20
USE_GPU = True
RUN_PARALELL = False
USE_RENDER = False
PRECOMP_FRAC = 0.3

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

parser = argparse.ArgumentParser()

parser.add_argument('--field-len', dest='field_len', type=int, default=FIELD_LENGTH, help='Length of the field.')
parser.add_argument('--tests', type=int, default=N_TESTS, help='Number of tests to run')
parser.add_argument('--max-foods', type=int, default=MAX_FOODS, help='Maximum number of foods.')
parser.add_argument('--spawn-foods', type=int, default=MAX_SPAWN_FOODS, help='Number of foods to spawn')
parser.add_argument('--data-dir', type=str, default='', help='Data directory')
parser.add_argument('--models-dir', type=str, default='', help='Model directory')
parser.add_argument('--logs-dir', type=str, default='', help='Logs directory')
parser.add_argument('--mode', type=int, default=TEST_MODE, choices=[0, 1, 2, 3],
						help='Team composition mode:'
							 '\n\t0 - Optimal agent controls interaction with an optimal follower '
							 '\n\t1 - Legible agent controls interaction with a legible follower '
							 '\n\t2 - Legible agent controls interaction with an optimal follower'
							 '\n\t3 - Optimal agent controls interaction with a legible follower')
parser.add_argument('--leg-agents', type=int, default=N_LEG_AGENTS, help='Number of legible agents')
parser.add_argument('--steps', type=int, default=STEPS_EPISODE, help='Maximum number of steps per episode')
parser.add_argument('--render', action='store_true', help='Flag that denotes the usage of a render')
parser.add_argument('--paralell', action='store_true', help='Flag that denotes the usage of a render')
parser.add_argument('--start-run', dest='start_run', type=int, default=0, help='Starting test run number')

input_args = parser.parse_args()
field_len = input_args.field_len
n_tests = input_args.tests
max_foods = input_args.max_foods
max_spawn_foods = input_args.spawn_foods
data_dir = input_args.data_dir
models_dir = input_args.models_dir
logs_dir = input_args.logs_dir
mode = input_args.mode
n_leg_agents = input_args.leg_agents
steps = input_args.steps
render = input_args.render or USE_RENDER
paralell = input_args.paralell or RUN_PARALELL
start_run = input_args.start_run

args = (' --mode %d --runs %d --n-agents %d --player-level %d --field-size %d --start-run %d'
        ' --n-food %d --food-level %d --steps-episode %d --n-foods-spawn %d --n-leg-agents %d --architecture %s --gamma %f'
        % (mode, n_tests, N_AGENTS, PLAYER_LVL, field_len, start_run, max_foods, FOOD_LEVEL, steps, max_spawn_foods, n_leg_agents, ARCHITECTURE, GAMMA))
args += ((' --render' if render else '') + (' --paralell' if paralell else '') + (' --use_gpu' if USE_GPU else '') + (" --fraction %f" % PRECOMP_FRAC) +
         (' --cnn' if USE_CNN else '') + (' --dueling' if USE_DUELING else '') + (' --ddqn' if USE_DDQN else '') + (' --vdn' if USE_VDN else '') +
         (' --models-dir %s' % models_dir if models_dir != '' else '') + (' --data-dir %s' % data_dir if data_dir != '' else '') +
         (' --logs-dir %s' % logs_dir if logs_dir != '' else ''))

command = "python " + str(tests_dir / 'test_lb_legible_collaboration.py') + args
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
