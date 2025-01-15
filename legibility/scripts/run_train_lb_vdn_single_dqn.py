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
BUFFER = 2000
GAMMA = 0.9
USE_DUELING = True
USE_DDQN = True
USE_CNN = True
USE_VDN = True
USE_TENSORBOARD = True

# Train params
MAX_CYCLES = 100
N_ITERATIONS = 400
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 10
ONLINE_LR = 0.001
TARGET_LR = 0.1
INIT_EPS = 1.0
FINAL_EPS = 0.05
# EPS_DECAY = 0.5	# for linear eps
# EPS_DECAY = 0.25	# for log eps
EPS_DECAY = 0.175	# for log eps
CYCLE_EPS = 0.991
# CYCLE_EPS = 1.0
EPS_TYPE = "log"
CYCLE_TYPE = "log"
USE_GPU = True
RESTART = False
DEBUG = False
RESTART_INFO = ["20230724-171745", "food_5x4_cycle_2", 2]
PRECOMP_FRAC = 0.3
TRAIN_VERSION = 'v1'

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
parser.add_argument('--batch-size', dest='batch_size', type=int, required=False, default=BATCH_SIZE)
parser.add_argument('--buffer-method', dest='buffer_method', type=str, required=False, default='uniform', choices=['uniform', 'weighted'],
					help='Method of deciding how to add new experience samples when replay buffer is full')
parser.add_argument('--buffer-smart-add', dest='buffer_smart_add', action='store_true',
					help='Flag denoting the use of smart sample add to experience replay buffer instead of first-in first-out')
parser.add_argument('--buffer-size', dest='buffer_size', type=int, required=False, default=BUFFER)
parser.add_argument('--cycle-eps', dest='cycle_eps', type=float, required=False, default=CYCLE_EPS, help='Epsilon decay for each cycle.')
parser.add_argument('--cycle-type', dest='cycle_type', type=str, required=False, default=CYCLE_TYPE, help='Type of decay for each cycle starting epsilon.')
parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
					help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=EPS_DECAY, help='Epsilon decay.')
parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, choices=['linear', 'log', 'exp', 'epoch'], default=EPS_TYPE,
                    help='Type of epsilon decay.')
parser.add_argument('--episode-steps', dest='max_steps', type=int, required=False, default=STEPS_EPISODE, help='Maximum number of steps per episode.')
parser.add_argument('--field-len', dest='field_len', type=int, required=False, default=FIELD_LENGTH, help='Length of the field.')
parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=FINAL_EPS, help='Minimum epsilon greedy.')
parser.add_argument('--iterations', dest='max_iterations', type=int, required=False, default=N_ITERATIONS, help='Number of iterations to train.')
parser.add_argument('--limits', dest='limits', nargs=2, type=int, required=False, default=[1, MAX_SPAWN_FOODS], help='Min and max number of food spawns to train.')
parser.add_argument('--tracker-dir', dest='logs', type=str, required=False, default='', help='Directory to store the performance logs.')
parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
					help='Directory to store trained models and load optimal models, if left blank stored in default location')
parser.add_argument('--no-force-coop', dest='no_force_coop', action='store_true', help='Flag denoting that the agents do not need to pick all items in full cooperation')
parser.add_argument('--online-lr', dest='online_lr', type=float, default=ONLINE_LR, help='Learning rate for the online model.')
parser.add_argument('--start-eps', dest='start_eps', type=float, required=False, default=INIT_EPS, help='Starting value for exploration epsilon greedy.')
parser.add_argument('--target-lr', dest='target_lr', type=float, default=TARGET_LR, help='Learning rate for the target model.')
parser.add_argument('--train-thresh', dest='train_thresh', type=float, required=False, default=None, help='Minimum performance threshold to skip model training.')
parser.add_argument('--use-lower-curriculum', dest='use_lower_model', action='store_true',
					help='Flag that signals the use of curriculum learning with a model with one less food item spawned.')
parser.add_argument('--use-higher-curriculum', dest='use_higher_model', action='store_true',
					help='Flag that signals the use of curriculum learning with a model with one more food item spawned.')
parser.add_argument('--version', dest='version', type=str, required=False, default=TRAIN_VERSION, choices=['v1', 'v2', 'v3'],
                    help='Model of the train script to use:\n\t- version 1: the food configuration changes in cycles that run for N iterations'
						 '\n\t- version 1: for each food item one model is trained and the food configuration changes every cycle, each cycle is composed by N iterations'
                         '\n\t- version 2: for each food item one model is trained and the food configuration changes every iteration, there are no cycles'
						 '\n\t- version 3: one model is trained for all food items and the food configuration changes every iteration, there are no cycles')
parser.add_argument('--warmup', dest='warmup', type=int, default=WARMUP_STEPS, help='Number of steps to collect data before starting train')

input_args = parser.parse_args()
add_method = input_args.buffer_method
buffer_size = input_args.buffer_size
batch_size = input_args.batch_size
cycle_type = input_args.cycle_type
cycle_eps = input_args.cycle_eps
data_dir = input_args.data_dir
eps_type = input_args.eps_type
eps_decay = input_args.eps_decay
field_len = input_args.field_len
final_eps = input_args.final_eps
iterations = input_args.max_iterations
limits = input_args.limits
logs_dir = input_args.logs_dir
models_dir = input_args.models_dir
max_steps = input_args.max_steps
no_force_coop = input_args.no_force_coop
online_lr = input_args.online_lr
smart_add = input_args.buffer_smart_add
start_eps = input_args.start_eps
target_lr = input_args.target_lr
tracker_logs = input_args.logs
train_thresh = input_args.train_thresh
train_version = input_args.version
use_lower_model = input_args.use_lower_model
use_higher_model = input_args.use_higher_model
warmup = input_args.warmup

for i in (reversed(range(limits[0], limits[1] + 1)) if use_higher_model else range(limits[0], limits[1] + 1)):
	print('Launching training script for %d foods spawned' % i)
	N_SPAWN_FOODS = i
	args = (" --nagents %d --architecture %s --buffer %d --gamma %f --iterations %d --max-cycles %d --batch %d --train-freq %d "
			"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --cycle-eps-decay %f "
			"--player-level %d --field-size %d --n-food %d --food-level %d --steps-episode %d --n-foods-spawn %d "
			% (N_AGENTS, ARQUITECTURE, buffer_size, GAMMA,                                                                                                      	# DQN parameters
			   iterations, MAX_CYCLES, batch_size, TRAIN_FREQ, TARGET_FREQ, online_lr, target_lr, start_eps, final_eps, eps_decay, eps_type, warmup, cycle_eps,  	# Train parameters
			   PLAYER_LEVEL, field_len, N_FOODS, FOOD_LVL, max_steps, N_SPAWN_FOODS)) 														                    	# Environment parameters
	args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
			 (" --cnn" if USE_CNN else "") + (" --tensorboard" if USE_TENSORBOARD else "") +
			 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else "") +
			 (" --debug" if DEBUG else "") + (" --vdn" if USE_VDN else "") + (" --fraction %f" % PRECOMP_FRAC) + (" --cycle-type %s" % cycle_type) +
			 (" --models-dir %s" % models_dir if models_dir != '' else "") + (" --data-dir %s" % data_dir if data_dir != '' else "") +
			 (" --logs-dir %s" % logs_dir if logs_dir != '' else "") + (" --use-lower-model" if use_lower_model else "") + (" --use-higher-model" if use_higher_model else "") +
			 (" --buffer-smart-add --buffer-method %s" % add_method if smart_add else "") + (" --tracker-dir %s" % tracker_logs if tracker_logs != '' else "") +
			 (" --train-performance %f" % train_thresh if train_thresh is not None else "") + (' --no-force-coop' if no_force_coop else ''))
	command = "python " + str(src_dir / ('train_lb_single_dqn%s.py' % ('_' + train_version if train_version != 'v1' else ''))) + args
	if not USE_SHELL:
		command = shlex.split(command)
		
	print(command)
	start_time = time.time()
	try:
		subprocess.run(command, shell=USE_SHELL, check=True)
	
	except subprocess.CalledProcessError as e:
		print(e.output)
		
	except KeyboardInterrupt as ki:
		print('Caught keyboard interrupt by user: %s Exiting....' % ki)
		break
		
	except Exception as e:
		print('Caught general exception: %s' % e)
		
	wall_time = time.time() - start_time
	print('Finished training, took %.3f seconds' % wall_time)

