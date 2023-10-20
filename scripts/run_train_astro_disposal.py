#! /usr/bin/env python
import shlex

import subprocess
from pathlib import Path

src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
log_dir = Path(__file__).parent.absolute().parent.absolute() / 'logs' / 'astro_disposal'
USE_SHELL = False

# DQN params
N_AGENTS = 2
N_LAYERS = 2
BUFFER = 1000
GAMMA = 0.95
LAYERS = (256, 256)
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_CNN = True
USE_TENSORBOARD = True

# Train params
N_ITERATIONS = 250
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 10
# ALPHA = 0.003566448247686571
ALPHA = 0.001
TAU = 0.1
INIT_EPS = 1.0
FINAL_EPS = 0.05
EPS_DECAY = 0.5	# for linear eps
# EPS_DECAY = 0.25	# for log eps
CYCLE_EPS = 0.97
EPS_TYPE = "linear"
USE_GPU = True
RESTART = False
DEBUG = True
RESTART_INFO = ["20230724-171745", "food_5x4_cycle_2", 2]
USE_RENDER = False

# Environment params
# GAME_LEVEL = ['level_one', 'level_two']
GAME_LEVEL = ['cramped_room']
STEPS_EPISODE = 400
WARMUP_STEPS = STEPS_EPISODE * 2
FIELD_LENGTH = 15
SLIP = False
FACING = True
AGENT_CENTERED = True
USE_ENCODING = True

args = (" --nagents %d --nlayers %d --buffer %d --gamma %f --layer-sizes %s --iterations %d --batch %d --train-freq %d "
		"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f "
		"--game-levels %s --max-env-steps %d --field-size %d %d "
		"--tensorboardDetails %s %d %d %s"
		% (N_AGENTS, N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]),																# DQN parameters
		   N_ITERATIONS, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS, CYCLE_EPS,	# Train parameters
		   ' '.join(GAME_LEVEL), STEPS_EPISODE, FIELD_LENGTH, FIELD_LENGTH,  																	# Environment parameters
		   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
		 (" --cnn" if USE_CNN else "") + (" --tensorboard" if USE_TENSORBOARD else "") +
		 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else "") +
		 (" --debug" if DEBUG else "") + (" --has-slip" if SLIP else "") + (" --force-facing" if FACING else "") +
		 (" --agent-centered" if AGENT_CENTERED else "") + (" --use-encoding" if USE_ENCODING else ""))
commamd = "python " + str(src_dir / 'train_astro_disposal_dqn.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)

print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
