#! /usr/bin/env python
import shlex
import time
import subprocess

from pathlib import Path


tests_dir = Path(__file__).parent.absolute().parent.absolute() / 'tests'
USE_SHELL = False

# DQN params
N_AGENTS = 2
N_LAYERS = 2
BUFFER = 10000
GAMMA = 0.9
LAYERS = (256, 256)
AGENTS = ("agent_1", "agent_2")
USE_DUELING = True
USE_DDQN = True
USE_CNN = True

# Environment params
N_FOODS = 8
FIELD_LENGTH = 8
STEPS_EPISODE = 400
PLAYER_LEVEL = 1
FOOD_LVL = 2
USE_RENDER = False
USE_GPU = False

# Testing params
N_CYCLES = 100
# MODEL_INFO = ["to_test", "all_final"]
TEST_MODE = 1
RUN_PARALLEL = False

models_dir = (Path(__file__).parent.absolute().parent.absolute() / 'models' / 'lb_coop_single_dqn' / ('%dx%d-field' % (FIELD_LENGTH, FIELD_LENGTH)) /
			  ('%d-agents' % N_AGENTS))
for i in range(1, N_FOODS + 1):
	N_SPAWN_FOODS = i
	models = (models_dir / ('%d-foods_%d-food-level' % (N_SPAWN_FOODS, FOOD_LVL))).iterdir()
	for model in models:
		MODEL_INFO = [str(model.name), "all_final"]
		args = (" --nagents %d --nlayers %d --buffer %d --gamma %f --layer-sizes %s --agent-ids %s "
				"--player-level %d --field-size %d --n-food %d --food-level %d --steps-episode %d "
				"--cycles %d --model-info %s %s --n-foods-spawn %d --test-mode %d --test-len %d"
				% (N_AGENTS, N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]), ' '.join([str(x) for x in AGENTS]),				# DQN parameters
				   PLAYER_LEVEL, FIELD_LENGTH, N_FOODS, FOOD_LVL, STEPS_EPISODE,															# Environment parameters
				   N_CYCLES, MODEL_INFO[0], MODEL_INFO[1], N_SPAWN_FOODS, TEST_MODE, N_SPAWN_FOODS))										# Testing parameters
		args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") +
				 (" --gpu" if USE_GPU else "") + (" --cnn" if USE_CNN else "") + (" --parallel" if RUN_PARALLEL else ""))
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
