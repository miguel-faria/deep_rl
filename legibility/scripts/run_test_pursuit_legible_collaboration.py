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
N_HUNTERS = 2
N_PREYS = 4
HUNTER_CLASSES = 1
N_REQUIRED_HUNTER = 2
FIELD_LENGTH = 10
STEPS_EPISODE = 600
PREY_TYPE = 'idle'

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
parser.add_argument('--catch-reward', dest='catch_reward', type=float, required=False, default=5.0, help='Reward for catching a prey')
parser.add_argument('--hunters', dest='n_hunters', type=int, required=False, default=N_HUNTERS, help='Number of hunters to spawn')
parser.add_argument('--preys', dest='n_preys', type=int, required=False, default=N_PREYS, help='Minimum and maximum number of preys')
parser.add_argument('--prey-type', dest='prey_type', type=str, required=False, choices=['idle', 'random', 'greedy'], default=PREY_TYPE,
                    help='Type of prey to be caught')
parser.add_argument('--required-hunters', dest='n_required_hunters', type=int, required=False, default=N_REQUIRED_HUNTER, help='Number of hunters to spawn')
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
catch_reward = input_args.catch_reward
n_hunters = input_args.n_hunters
n_required_hunters = input_args.n_required_hunters
prey_type = input_args.prey_type
n_preys = input_args.n_preys
data_dir = input_args.data_dir
models_dir = input_args.models_dir
logs_dir = input_args.logs_dir
mode = input_args.mode
n_leg_agents = input_args.leg_agents
steps = input_args.steps
render = input_args.render or USE_RENDER
paralell = input_args.paralell or RUN_PARALELL
start_run = input_args.start_run

hunter_ids = ' '.join([('h%d' % (idx + 1)) for idx in range(n_hunters)])
prey_ids = ' '.join([('p%d' % (idx + 1)) for idx in range(n_preys)])

args = (' --mode %d --runs %d --field-size %d --start-run %d'
		' --hunter-ids %s --prey-ids %s --hunter-classes %d --prey-type %s --n-hunters-catch %d --catch-reward %f'
        ' --steps-episode %d --n-leg-agents %d --architecture %s --gamma %f'
		% (mode, n_tests, field_len, start_run, hunter_ids, prey_ids, HUNTER_CLASSES, prey_type, n_required_hunters, catch_reward, steps, n_leg_agents, ARCHITECTURE, GAMMA))
args += ((' --render' if render else '') + (' --paralell' if paralell else '') + (' --use_gpu' if USE_GPU else '') + (" --fraction %f" % PRECOMP_FRAC) +
         (' --cnn' if USE_CNN else '') + (' --dueling' if USE_DUELING else '') + (' --ddqn' if USE_DDQN else '') + (' --vdn' if USE_VDN else '') +
         (' --models-dir %s' % models_dir if models_dir != '' else '') + (' --data-dir %s' % data_dir if data_dir != '' else '') +
         (' --logs-dir %s' % logs_dir if logs_dir != '' else ''))

command = "python " + str(tests_dir / 'test_pursuit_legible_collaboration.py') + args
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
