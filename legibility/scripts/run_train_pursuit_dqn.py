#! /usr/bin/env python
import shlex

import subprocess
from pathlib import Path


src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'pursuit'
USE_SHELL = False

# DQN params
N_LAYERS = 2
BUFFER = 1000
GAMMA = 0.95
LAYERS = (128, 128)
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = False
USE_DDQN = True

# Train params
CYCLES = 1
N_ITERATIONS = 250
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 1
ALPHA = 0.001
TAU = 0.01
INIT_EPS = 1.0
FINAL_EPS = 0.1
EPS_DECAY = 0.5
EPS_TYPE = "linear"

# Environment params
HUNTERS = ['hunter_1', 'hunter_2']
PREYS = ['prey_1_train']
FIELD_LENGTH = 10
STEPS_EPISODE = 300
MIN_CATCH = 2
WARMUP_STEPS = STEPS_EPISODE * 2

args = (" --nlayers %d --buffer %d --gamma %f --layer-sizes %s --gpu --cycles %d --iterations %d --batch %d --train-freq %d "
		"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d "
		"--field-size %d --hunters %s --preys %s --n-catch %d --steps-episode %d --tensorboard --tensorboardDetails %s %d %d %s"
		% (N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]),  																	# DQN parameters
		   CYCLES, N_ITERATIONS, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS,  	# Train parameters
		   FIELD_LENGTH, ' '.join(HUNTERS), ' '.join(PREYS), MIN_CATCH, STEPS_EPISODE,  													# Environment parameters
		   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
args += (" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "")
commamd = "python " + str(src_dir / 'train_pursuit_dqn.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)
	
print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
