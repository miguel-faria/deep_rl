#! /usr/bin/env python
import shlex
import time
import subprocess
import argparse

from pathlib import Path


tests_dir = Path(__file__).parent.absolute().parent.absolute() / 'tests'
USE_SHELL = False

# DQN params
N_AGENTS = 2
ARQUITECTURE = "v3"
GAMMA = 0.9
AGENTS = ("agent_1", "agent_2")
USE_DUELING = True
USE_DDQN = True
USE_CNN = True
USE_VDN = True

# Environment params
N_FOODS = 8
FIELD_LENGTH = 8
STEPS_EPISODE = 400
PLAYER_LEVEL = 1
FOOD_LVL = 2
USE_RENDER = False
USE_GPU = True
DEBUG = False

# Testing params
N_CYCLES = 250
TEST_MODE = 0
RUN_PARALLEL = False
N_SPAWN_FOODS = N_FOODS
MODEL_INFO = ["best", "all_final"]


models_dir = (Path(__file__).parent.absolute().parent.absolute() / 'models' / ('lb_coop_single%s_dqn' % ('_vdn' if USE_VDN else '')) /
			  ('%dx%d-field' % (FIELD_LENGTH, FIELD_LENGTH)) / ('%d-agents' % N_AGENTS))
parser = argparse.ArgumentParser()
parser.add_argument('--limits', dest='limits', nargs=2, type=int, required=False, default=[1, N_SPAWN_FOODS],
					help='Minimum and maximum food spawns')
args = parser.parse_args()

for i in range(args.limits[0], args.limits[1] + 1):
	n_spawn_foods = i
	models = [m.name for m in (models_dir / ('%d-foods_%d-food-level' % (n_spawn_foods, FOOD_LVL))).iterdir()]
	args = (" --nagents %d --architecture %s --gamma %f --agent-ids %s "
					"--player-level %d --field-size %d --n-food %d --food-level %d --steps-episode %d "
					"--cycles %d --model-info %s %s --n-foods-spawn %d --test-mode %d --test-len %d"
			% (N_AGENTS, ARQUITECTURE, GAMMA, ' '.join([str(x) for x in AGENTS]),														# DQN parameters
			   PLAYER_LEVEL, FIELD_LENGTH, N_FOODS, FOOD_LVL, STEPS_EPISODE,															# Environment parameters
			   N_CYCLES, MODEL_INFO[0], MODEL_INFO[1], n_spawn_foods, TEST_MODE, n_spawn_foods))										# Testing parameters
	args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") +
			 (" --gpu" if USE_GPU else "") + (" --cnn" if USE_CNN else "") + (" --parallel" if RUN_PARALLEL else "") +
			 (" --debug" if DEBUG else "") + (" --vdn" if USE_VDN else ""))
	commamd = "python " + str(tests_dir / 'test_lb_single_dqn_model.py') + args
	if not USE_SHELL:
		commamd = shlex.split(commamd)
		
	print(commamd)
	start_time = time.time()
	try:
		subprocess.run(commamd, shell=USE_SHELL, check=True)
	
	except subprocess.CalledProcessError as e:
		print(e.output)
		
	except KeyboardInterrupt as ki:
		print('Caught keyboard interrupt by user: %s Exiting....' % ki)
		
	except Exception as e:
		print('Caught general exception: %s' % e)
		
	wall_time = time.time() - start_time
	print('Finished testing, took %.3f seconds' % wall_time)
