#! /usr/bin/env python
import shlex
import subprocess
import time

from pathlib import Path


src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'pursuit'
USE_SHELL = False

# DQN params
N_AGENTS = 2
N_LAYERS = 2
BUFFER = 1000
GAMMA = 0.95
LAYERS = (128, 128)
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_CNN = True
USE_TENSORBOARD = True

# Train params
CYCLES = 4
# N_ITERATIONS = 1500
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 1
ALPHA = 0.0001
TAU = 0.001
INIT_EPS = 1.0
FINAL_EPS = 0.05
EPS_DECAY = 0.1
# EPS_DECAY = 0.5	# for linear eps
CYCLE_EPS = 0.97
EPS_TYPE = "log"
USE_GPU = True
RESTART = False
DEBUG = False
RESTART_INFO = ["20230724-171745", "food_5x4_cycle_2", 2]

# Environment params
N_PREYS = 1
HUNTER_IDS = [('hunter_%d' % (idx + 1)) for idx in range(N_AGENTS)]
PREY_IDS = [('prey_%d' % (idx + 1)) for idx in range(N_PREYS)]
HUNTER_CLASSES = 1
N_REQUIRED_HUNTER = 2
FIELD_LENGTH = 10
STEPS_EPISODE = 500
WARMUP_STEPS = STEPS_EPISODE * 2
USE_RENDER = False
USE_TARGETS = False
TRAIN_TARGETS = ['prey_1']

for iterations in range(1000, 3000, 250):
	args = (" --nagents %d --nlayers %d --buffer %d --gamma %f --layer-sizes %s --iterations %d --batch %d --train-freq %d "
			"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f "
			"--hunter-ids %s --prey-ids %s --hunter-classes %d --field-size %d --n-hunters-catch %d --steps-episode %d --tensorboardDetails %s %d %d %s"
			% (N_AGENTS, N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]),															# DQN parameters
			   iterations, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS, CYCLE_EPS,	# Train parameters
			   ' '.join(HUNTER_IDS), ' '.join(PREY_IDS), HUNTER_CLASSES, FIELD_LENGTH, N_REQUIRED_HUNTER, STEPS_EPISODE,						# Environment parameters
			   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
	args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
			 (" --cnn" if USE_CNN else "") + (" --tensorboard" if USE_TENSORBOARD else "") +
			 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else "") +
			 (" --debug" if DEBUG else "") + ((" --train-targets %s" % " ".join(TRAIN_TARGETS)) if USE_TARGETS else ""))
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

