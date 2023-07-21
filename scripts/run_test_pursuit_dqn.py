#! /usr/bin/env python
import shlex

import subprocess
from pathlib import Path


tests_dir = Path(__file__).parent.absolute().parent.absolute() / 'tests'
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

# Environment params
HUNTERS = ['hunter_1', 'hunter_2']
PREYS = ['prey_1']
FIELD_LENGTH = 10
STEPS_EPISODE = 300
MIN_CATCH = 2
WARMUP_STEPS = STEPS_EPISODE * 2

args = (" --nlayers %d --buffer %d --gamma %f --layer-sizes %s --gpu --tensorboard --tensorboardDetails %s %d %d %s "
		"--field-size %d --hunters %s --preys %s --n-catch %d --steps-episode %d"
		% (N_LAYERS, BUFFER, GAMMA, ' '.join([str(x) for x in LAYERS]), TENSORBOARD_DATA[0], TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3],  # DQN parameters
		   FIELD_LENGTH, ' '.join(HUNTERS), ' '.join(PREYS), MIN_CATCH, STEPS_EPISODE)  # Environment parameters
		   )
args += (" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "")
commamd = "python " + str(tests_dir / 'test_pursuit_dqn_model.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)

print(commamd)
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)
