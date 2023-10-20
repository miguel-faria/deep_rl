#! /usr/bin/env python
import shlex
import subprocess
import time

from pathlib import Path


src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'lb_foraging'
USE_SHELL = False

# DQN params
N_AGENTS = 2
N_LAYERS = 2
BUFFER = 1000
GAMMA = 0.9
LAYERS = (256, 256)
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_TENSORBOARD = True

# Train params
# CYCLES = 4
N_ITERATIONS = 250
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 1
# ALPHA = 0.003566448247686571
ALPHA = 0.001
TAU = 0.01
INIT_EPS = 1.0
FINAL_EPS = 0.1
EPS_DECAY = 0.5
CYCLE_EPS = 0.93
EPS_TYPE = "linear"
USE_GPU = True
RESTART = False
RESTART_INFO = ["20230724-171745", "food_5x4_cycle_2", 2]

# Environment params
N_FOODS = 8
FIELD_LENGTH = 8
STEPS_EPISODE = 400
PLAYER_LEVEL = 1
FOOD_LVL = 2
WARMUP_STEPS = STEPS_EPISODE * 2
N_SPAWN_FOODS = 8
USE_RENDER = False

args = (" --nagents %d --nlayers %d --buffer %d --gamma %f --layer-sizes %s --iterations %d --batch %d --train-freq %d "
		"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f "
		"--player-level %d --field-size %d --n-food %d --food-level %d --steps-episode %d --n-foods-spawn %d --tensorboardDetails %s %d %d %s"
		% (N_AGENTS, N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]),															# DQN parameters
		   N_ITERATIONS, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS, CYCLE_EPS,# Train parameters
		   PLAYER_LEVEL, FIELD_LENGTH, N_FOODS, FOOD_LVL, STEPS_EPISODE, N_SPAWN_FOODS,														# Environment parameters
		   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
		 (" --tensorboard" if USE_TENSORBOARD else "") +
		 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else ""))
commamd = "python " + str(src_dir / 'train_lb_centralized_madqn.py') + args
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
print('Finished training, took %.3f seconds' % wall_time)
