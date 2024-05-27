#! /usr/bin/env python
import shlex
import subprocess
import time
import argparse

from pathlib import Path


src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'pursuit'
USE_SHELL = False

# DQN params
N_AGENTS = 2
BUFFER = 10000
GAMMA = 0.9
ARQUITECTURE = "v3"
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_VDN = True
USE_CNN = True
USE_TENSORBOARD = True

# Train params
CYCLES = 2
N_ITERATIONS = 1000
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 10
ALPHA = 0.001
TAU = 0.1
INIT_EPS = 1.0
FINAL_EPS = 0.1
EPS_DECAY = 0.08	# for log eps
# EPS_DECAY = 0.5	# for linear eps
CYCLE_EPS = 0.97
EPS_TYPE = "log"
USE_GPU = True
RESTART = False
DEBUG = False
RESTART_INFO = ["20230724-171745", "food_5x4_cycle_2", 2]
PRECOMP_FRAC = 0.3

# Environment params
MAX_PREYS = 4
HUNTER_CLASSES = 1
N_REQUIRED_HUNTER = 2
FIELD_LENGTH = 10
STEPS_EPISODE = 400
WARMUP_STEPS = STEPS_EPISODE * 2
USE_RENDER = False
USE_TARGETS = False
TRAIN_TARGETS = ['p1']
PREY_TYPE = 'idle'

parser = argparse.ArgumentParser()
parser.add_argument('--limits', dest='limits', nargs=2, type=int, required=False, default=[1, MAX_PREYS],
					help='Minimum and maximum number of preys')
parser.add_argument('--catch_reward', dest='catch_reward', type=int, required=False, default=5.0,
					help='Reward for catching a prey')
parser.add_argument('--prey-type', dest='prey_type', type=str, required=False, choices=['idle', 'random', 'greedy'], default=PREY_TYPE,
					help='Type of prey to be caught')
parser.add_argument('--field', dest='field', type=int, required=False, default=FIELD_LENGTH,
					help='Length of the field of the environment')
parser.add_argument('--hunters', dest='n_hunters', type=int, required=False, default=N_AGENTS,
					help='Number of hunters to spawn')
parser.add_argument('--iterations', dest='n_iterations', type=int, required=False, default=N_ITERATIONS,
					help='Number of iterations per training cycle')


input_args = parser.parse_args()

for i in range(input_args.limits[0], input_args.limits[1] + 1):
	n_preys = i
	n_hunters = input_args.n_hunters
	n_iterations = input_args.n_iterations
	HUNTER_IDS = [('h%d' % (idx + 1)) for idx in range(n_hunters)]
	PREY_IDS = [('p%d' % (idx + 1)) for idx in range(n_preys)]
	prey_type = input_args.prey_type
	field_length = input_args.field
	args = (" --nagents %d --architecture %s --buffer %d --gamma %f --iterations %d --batch %d --train-freq %d "
			"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f "
			"--hunter-ids %s --prey-ids %s --hunter-classes %d --prey-type %s --field-size %d --n-hunters-catch %d --steps-episode %d --catch-reward %f "
			"--tensorboardDetails %s %d %d %s"
			% (n_hunters, ARQUITECTURE, BUFFER, GAMMA,																												# DQN parameters
			   n_iterations, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS, CYCLE_EPS,					# Train parameters
			   ' '.join(HUNTER_IDS), ' '.join(PREY_IDS), HUNTER_CLASSES, prey_type, field_length, N_REQUIRED_HUNTER, STEPS_EPISODE, input_args.catch_reward,		# Environment parameters
			   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
	args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
			 (" --cnn" if USE_CNN else "") + (" --tensorboard" if USE_TENSORBOARD else "") + (" --vdn" if USE_VDN else "") +
			 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else "") +
			 (" --debug" if DEBUG else "") + ((" --train-targets %s" % " ".join(TRAIN_TARGETS)) if USE_TARGETS else "") + (" --fraction %f" % PRECOMP_FRAC))
	commamd = "python " + str(src_dir / 'train_pursuit_single_dqn.py') + args
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

