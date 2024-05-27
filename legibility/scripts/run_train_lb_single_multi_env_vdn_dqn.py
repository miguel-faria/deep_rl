#! /usr/bin/env python
import shlex
import subprocess
import time
import argparse

from pathlib import Path


src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'lb_foraging'
USE_SHELL = False

# DQN params
N_AGENTS = 2
ARQUITECTURE = "v3"
BUFFER = 12500
GAMMA = 0.9
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_CNN = True
USE_VDN = True
USE_TENSORBOARD = True

# Train params
MAX_CYCLES = 100
N_ITERATIONS = 400
BATCH_SIZE = 32
TRAIN_FREQ = 5
TARGET_FREQ = 20
# ALPHA = 0.003566448247686571
ALPHA = 0.001
TAU = 0.075
INIT_EPS = 1.0
FINAL_EPS = 0.05
# EPS_DECAY = 0.5	# for linear eps
EPS_DECAY = 0.25	# for log eps
# CYCLE_EPS = 0.99
CYCLE_EPS = 1.0
EPS_TYPE = "log"
USE_GPU = True
RESTART = False
DEBUG = False
RESTART_INFO = ["20230724-171745", "food_5x4_cycle_2", 2]
PRECOMP_FRAC = 0.2

# Environment params
N_FOODS = 8
MAX_SPAWN_FOODS = 8
FIELD_LENGTH = 8
STEPS_EPISODE = 400
PLAYER_LEVEL = 1
FOOD_LVL = 2
WARMUP_STEPS = STEPS_EPISODE * 2
USE_RENDER = False

parser = argparse.ArgumentParser()
parser.add_argument('--limits', dest='limits', nargs=2, type=int, required=False, default=[1, MAX_SPAWN_FOODS],
					help='Minimum and maximum food spawns')
parser.add_argument('--field-len', dest='field_len', type=int, required=False, default=FIELD_LENGTH,
					help='Length of the field')
parser.add_argument('--episode-steps', dest='max_steps', type=int, required=False, default=STEPS_EPISODE)
parser.add_argument('--iterations', dest='max_iterations', type=int, required=False, default=N_ITERATIONS)
parser.add_argument('--tags', dest='tags', type=str, nargs='+', required=False, default=None)
args = parser.parse_args()
field_len = args.field_len
limits = args.limits
max_steps = args.max_steps
iterations = args.max_iterations
tags = args.tags if args.tags is not None else None

for i in range(limits[0], limits[1] + 1):
	print('Launching training script for %d foods spawned' % i)
	N_SPAWN_FOODS = i
	args = (" --nagents %d --architecture %s --buffer %d --gamma %f --iterations %d --max-cycles %d --batch %d --train-freq %d "
			"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f "
			"--player-level %d --field-size %d --n-food %d --food-level %d --steps-episode %d --n-foods-spawn %d --tensorboardDetails %s %d %d %s"
			% (N_AGENTS, ARQUITECTURE, BUFFER, GAMMA,																										# DQN parameters
			   iterations, MAX_CYCLES, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS, CYCLE_EPS,	# Train parameters
			   PLAYER_LEVEL, field_len, N_FOODS, FOOD_LVL, max_steps, N_SPAWN_FOODS,																		# Environment parameters
			   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
	args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
			 (" --cnn" if USE_CNN else "") + (" --tensorboard" if USE_TENSORBOARD else "") +
			 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else "") +
			 (" --debug" if DEBUG else "") + (" --vdn" if USE_VDN else "") + (" --fraction %f" % PRECOMP_FRAC) +
			 (" --train-tags %s" % (' '.join(tags))) if tags is not None else "")
	commamd = "python " + str(src_dir / 'train_lb_single_multi_env_dqn.py') + args
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
		break
		
	except Exception as e:
		print('Caught general exception: %s' % e)
		
	wall_time = time.time() - start_time
	print('Finished training, took %.3f seconds' % wall_time)

