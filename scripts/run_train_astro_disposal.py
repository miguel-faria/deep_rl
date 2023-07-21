#! /usr/bin/env python
import shlex

import subprocess
from pathlib import Path

src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'astro_disposal'
USE_SHELL = False

# DQN params
N_LAYERS = 2
BUFFER = 1000
GAMMA = 0.95
LAYERS = (128, 128)
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = False
USE_DDQN = True
USE_TENSORBOARD = True

# Train params
N_ITERATS = 5000
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 1
ALPHA = 0.001
TAU = 0.01
INIT_EPS = 1.0
FINAL_EPS = 0.05
EPS_DECAY = 0.5
EPS_TYPE = "linear"
USE_GPU = True

# Environment params
# GAME_LEVEL = ['level_one', 'level_two']
GAME_LEVEL = ['level_one']
STEPS_EPISODE = 400
WARMUP_STEPS = STEPS_EPISODE * 2

args = (" --nlayers %d --buffer %d --gamma %f --layer-sizes %s --iterations %d --batch %d --train-freq %d "
		"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --game-level %s --max-env-steps %d "
		"--tensorboardDetails %s %d %d %s"
		% (N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]),  														# DQN parameters
		   N_ITERATS, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS,  # Train parameters
		   ' '.join(GAME_LEVEL), STEPS_EPISODE,  																					# Environment parameters
		   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + ("  --gpu" if USE_GPU else "") +
		(" --tensorboard" if USE_TENSORBOARD else ""))
commamd = "python " + str(src_dir / 'train_astro_disposal_dqn.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)

print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
