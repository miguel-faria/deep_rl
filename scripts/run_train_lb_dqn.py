#! /usr/bin/env python
import shlex

import subprocess
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
AGENTS = ("agent_1", "agent_2")
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_TENSORBOARD = True

# Train params
CYCLES = 4
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
EPS_TYPE = "linear"
USE_GPU = True
RESTART = False
RESTART_INFO = ["20230721-111816", "food_5x4_cycle_2", 2]

# Environment params
N_FOODS = 8
FIELD_LENGTH = 8
STEPS_EPISODE = 400
PLAYER_LEVEL = 1
FOOD_LVL = 2
WARMUP_STEPS = STEPS_EPISODE * 2
N_SPAWN_FOODS = 6
USE_RENDER = False

args = (" --nagents %d --nlayers %d --buffer %d --gamma %f --layer-sizes %s --agent-ids %s --cycles %d --iterations %d --batch %d --train-freq %d "
		"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --player-level %d --field-size %d "
		"--n-food %d --food-level %d --steps-episode %d --n-foods-spawn %d --tensorboardDetails %s %d %d %s"
		% (N_AGENTS, N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]), ' '.join([str(x) for x in AGENTS]),						# DQN parameters
		   CYCLES, N_ITERATIONS, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS,	# Train parameters
		   PLAYER_LEVEL, FIELD_LENGTH, N_FOODS, FOOD_LVL, STEPS_EPISODE, N_SPAWN_FOODS,														# Environment parameters
		   TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
		 (" --tensorboard" if USE_TENSORBOARD else "") +
		 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else ""))
commamd = "python " + str(src_dir / 'train_lb_dqn.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)
	
print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
