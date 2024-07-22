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
N_LEG_AGENTS = 1
ARQUITECTURE = "v3"
BUFFER = 2000
GAMMA = 0.95
BETA = 0.9
TEMP = 0.1
TENSORBOARD_DATA = [str(log_dir), 50, 25, '.log']
USE_DUELING = True
USE_DDQN = True
USE_CNN = True
USE_VDN = True
USE_TENSORBOARD = True
LEG_REWARD = 'q_vals'

# Train params
MAX_CYCLES = 100
N_ITERATIONS = 400
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 10
# ALPHA = 0.003566448247686571
ALPHA = 0.001
TAU = 0.1
INIT_EPS = 1.0
FINAL_EPS = 0.05
# EPS_DECAY = 0.7	# for linear eps
EPS_DECAY = 0.175	# for log eps
# CYCLE_EPS = 0.925
CYCLE_EPS = 0.3
EPS_TYPE = "log"
CYCLE_TYPE = "linear"
USE_GPU = True
RESTART = False
DEBUG = False
RESTART_INFO = ["20230724-171745", "food_5x4_cycle_2", 2]
OPT_VDN = True
PRECOMP_FRAC = 0.3

# Environment params
N_PLAYERS = 2
N_FOODS = 8
MAX_SPAWN_FOODS = 8
FIELD_LENGTH = 8
STEPS_EPISODE = 400
PLAYER_LEVEL = 1
FOOD_LVL = 2
WARMUP_STEPS = STEPS_EPISODE * 2
USE_RENDER = False

parser = argparse.ArgumentParser()
parser.add_argument('--limits', dest='limits', nargs=2, type=int, required=False, default=[1, MAX_SPAWN_FOODS], help='Min and max number of food spawns to train.')
parser.add_argument('--field-len', dest='field_len', type=int, required=False, default=FIELD_LENGTH, help='Length of the field.')
parser.add_argument('--episode-steps', dest='max_steps', type=int, required=False, default=STEPS_EPISODE, help='Maximum number of steps per episode.')
parser.add_argument('--iterations', dest='max_iterations', type=int, required=False, default=N_ITERATIONS, help='Number of iterations to train.')
parser.add_argument('--logs', dest='logs', type=str, required=False, default=TENSORBOARD_DATA[0], help='Directory to store the performance logs.')
parser.add_argument('--legible-reward', dest='legible_reward', type=str, choices=['simple', 'q_vals', 'info', 'reward'], required=False, default=LEG_REWARD,
					help='Type of legible reward. Types: simple, q_vals, info, reward')
parser.add_argument('--cycle-type', dest='cycle_type', type=str, required=False, default=CYCLE_TYPE, help='Type of decay for each cycle starting epsilon.')
parser.add_argument('--cycle-eps', dest='cycle_eps', type=float, required=False, default=CYCLE_EPS, help='Epsilon decay for each cycle.')
parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, default=EPS_TYPE, help='Type of epsilon decay.')
parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=EPS_DECAY, help='Epsilon decay.')
parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
					help='Directory to store trained models and load optimal models, if left blank stored in default location')
parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
					help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
parser.add_argument('--use-lower-curriculum', dest='use_lower_model', action='store_true',
					help='Flag that signals the use of curriculum learning with a model with one less food item spawned.')
parser.add_argument('--use-higher-curriculum', dest='use_higher_model', action='store_true',
					help='Flag that signals the use of curriculum learning with a model with one more food item spawned.')
parser.add_argument('--buffer-smart-add', dest='buffer_smart_add', action='store_true',
					help='Flag denoting the use of smart sample add to experience replay buffer instead of first-in first-out')
parser.add_argument('--buffer-method', dest='buffer_method', type=str, required=False, default='uniform', choices=['uniform', 'weighted'],
					help='Method of deciding how to add new experience samples when replay buffer is full')

input_args = parser.parse_args()
add_method = input_args.buffer_method
cycle_type = input_args.cycle_type
cycle_eps = input_args.cycle_eps
data_dir = input_args.data_dir
eps_type = input_args.eps_type
eps_decay = input_args.eps_decay
field_len = input_args.field_len
iterations = input_args.max_iterations
leg_reward = input_args.legible_reward
limits = input_args.limits
logs_dir = input_args.logs_dir
models_dir = input_args.models_dir
max_steps = input_args.max_steps
smart_add = input_args.buffer_smart_add
tracker_logs = input_args.logs
use_lower_model = input_args.use_lower_model
use_higher_model = input_args.use_higher_model

for i in (reversed(range(limits[0], limits[1] + 1)) if use_higher_model else range(limits[0], limits[1] + 1)):
	print('Launching training script for %d foods spawned' % i)
	N_SPAWN_FOODS = i
	if i == 1:
		eps = 'log'
		decay = 0.175
	else:
		eps = eps_type
		decay = eps_decay
	args = (" --n-agents %d --architecture %s --buffer %d --gamma %f --beta %f --reward %s --iterations %d --max-cycles %d --batch %d --train-freq %d "
			"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f --legibility-temp %f "
			"--n-players %d --player-level %d --field-size %d --n-food %d --food-level %d --steps-episode %d --n-foods-spawn %d --tensorboardDetails %s %d %d %s"
			% (N_AGENTS, ARQUITECTURE, BUFFER, GAMMA, BETA, leg_reward,  																					# DQN parameters
			   iterations, MAX_CYCLES, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ, ALPHA, TAU, INIT_EPS, FINAL_EPS, decay, eps, WARMUP_STEPS, cycle_eps, TEMP,  	# Train parameters
			   N_PLAYERS, PLAYER_LEVEL, field_len, N_FOODS, FOOD_LVL, max_steps, N_SPAWN_FOODS,  															# Environment parameters
			   tracker_logs, TENSORBOARD_DATA[1], TENSORBOARD_DATA[2], TENSORBOARD_DATA[3]))
	args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
			 (" --cnn" if USE_CNN else "") + (" --tensorboard" if USE_TENSORBOARD else "") + (" --vdn" if USE_VDN else "") +
			 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else "") +
			 (" --debug" if DEBUG else "") + (" --use-opt-vdn" if OPT_VDN else "") + (" --n-leg-agents %d" % N_LEG_AGENTS) + (" --fraction %f" % PRECOMP_FRAC) +
			 (" --models-dir %s" % models_dir if models_dir != '' else "") + (" --cycle-eps-type %s" % cycle_type)  + (" --data-dir %s" % data_dir if data_dir != '' else "") +
			 (" --logs-dir %s" % logs_dir if logs_dir != '' else "") + (" --use-lower-model" if use_lower_model else "") +  (" --use-higher-model" if use_higher_model else "") +
			 (" --buffer-smart-add --buffer-method %s" % add_method if smart_add else "") )
	commamd = "python " + str(src_dir / 'train_lb_legible_dqn.py') + args
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

