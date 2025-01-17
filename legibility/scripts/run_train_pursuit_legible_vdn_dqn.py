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
N_LEG_AGENTS = 1
ARQUITECTURE = "v3"
BUFFER = 10000
GAMMA = 0.9
BETA = 0.9
TEMP = 0.1
USE_DUELING = True
USE_DDQN = True
USE_VDN = True
USE_CNN = True
USE_TRACKER = True
LEG_REWARD = 'q_vals'

# Train params
N_ITERATIONS = 1000
BATCH_SIZE = 32
TRAIN_FREQ = 1
TARGET_FREQ = 10
ONLINE_LR = 0.001
TARGET_LR = 0.1
INIT_EPS = 1.0
FINAL_EPS = 0.05
EPS_DECAY = 0.08	# for log eps
# EPS_DECAY = 0.5	# for linear eps
EPS_TYPE = "log"
USE_GPU = True
RESTART = False
DEBUG = False
OPT_VDN = True
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
PREY_TYPE = 'idle'

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', dest='batch_size', type=int, required=False, default=BATCH_SIZE)
parser.add_argument('--beta', dest='beta', type=float, required=False, default=BETA, help='Value for adherence to the optimal policy. Default: 0.9')
parser.add_argument('--buffer-method', dest='buffer_method', type=str, required=False, default='uniform', choices=['uniform', 'weighted'],
					help='Method of deciding how to add new experience samples when replay buffer is full')
parser.add_argument('--buffer-smart-add', dest='buffer_smart_add', action='store_true',
					help='Flag denoting the use of smart sample add to experience replay buffer instead of first-in first-out')
parser.add_argument('--buffer-size', dest='buffer_size', type=int, required=False, default=BUFFER)
parser.add_argument('--catch-reward', dest='catch_reward', type=float, required=False, default=5.0, help='Reward for catching a prey')
parser.add_argument('--data-dir', dest='data_dir', type=str, default='',
					help='Directory to retrieve data regarding configs and model performances, if left blank using default location')
parser.add_argument('--eps-decay', dest='eps_decay', type=float, required=False, default=EPS_DECAY, help='Epsilon decay.')
parser.add_argument('--eps-type', dest='eps_type', type=str, required=False, choices=['linear', 'log', 'exp', 'epoch'], default=EPS_TYPE,
                    help='Type of epsilon decay.')
parser.add_argument('--episode-steps', dest='max_steps', type=int, required=False, default=STEPS_EPISODE, help='Maximum number of steps per episode.')
parser.add_argument('--field-len', dest='field_len', type=int, required=False, default=FIELD_LENGTH, help='Length of the field.')
parser.add_argument('--final-eps', dest='final_eps', type=float, required=False, default=FINAL_EPS, help='Minimum epsilon greedy.')
parser.add_argument('--iterations', dest='max_iterations', type=int, required=False, default=N_ITERATIONS, help='Number of iterations to train.')
parser.add_argument('--hunters', dest='n_hunters', type=int, required=False, default=N_AGENTS, help='Number of hunters to spawn')
parser.add_argument('--legible-reward', dest='legible_reward', type=str, choices=['simple', 'q_vals', 'info', 'reward'], required=False, default=LEG_REWARD,
					help='Type of legible reward. Types: simple, q_vals, info, reward')
parser.add_argument('--legibility-temp', dest='temp', type=float, required=False, default=TEMP, help='Value for temperature parameter that governs the legibility reward. Default: 0.9')
parser.add_argument('--limits', dest='limits', nargs=2, type=int, required=False, default=[1, MAX_PREYS], help='Min and max number of food spawns to train.')
parser.add_argument('--logs-dir', dest='logs_dir', type=str, default='', help='Directory to store logs, if left blank stored in default location')
parser.add_argument('--models-dir', dest='models_dir', type=str, default='',
					help='Directory to store trained models and load optimal models, if left blank stored in default location')
parser.add_argument('--online-lr', dest='online_lr', type=float, default=ONLINE_LR, help='Learning rate for the online model.')
parser.add_argument('--prey-type', dest='prey_type', type=str, required=False, choices=['idle', 'random', 'greedy'], default=PREY_TYPE,
                    help='Type of prey to be caught')
parser.add_argument('--start-eps', dest='start_eps', type=float, required=False, default=INIT_EPS, help='Starting value for exploration epsilon greedy.')
parser.add_argument('--target-lr', dest='target_lr', type=float, default=TARGET_LR, help='Learning rate for the target model.')
parser.add_argument('--tracker-dir', dest='logs', type=str, required=False, default='', help='Directory to store the performance logs.')
parser.add_argument('--train-thresh', dest='train_thresh', type=float, required=False, default=None, help='Minimum performance threshold to skip model training.')
parser.add_argument('--use-lower-curriculum', dest='use_lower_model', action='store_true',
					help='Flag that signals the use of curriculum learning with a model with one less food item spawned.')
parser.add_argument('--use-higher-curriculum', dest='use_higher_model', action='store_true',
					help='Flag that signals the use of curriculum learning with a model with one more food item spawned.')
parser.add_argument('--warmup', dest='warmup', type=int, default=WARMUP_STEPS, help='Number of steps to collect data before starting train')

input_args = parser.parse_args()
add_method = input_args.buffer_method
beta = input_args.beta
buffer_size = input_args.buffer_size
batch_size = input_args.batch_size
catch_reward = input_args.catch_reward
data_dir = input_args.data_dir
eps_type = input_args.eps_type
eps_decay = input_args.eps_decay
field_len = input_args.field_len
final_eps = input_args.final_eps
n_iterations = input_args.max_iterations
leg_reward = input_args.legible_reward
limits = input_args.limits
logs_dir = input_args.logs_dir
models_dir = input_args.models_dir
max_steps = input_args.max_steps
n_hunters = input_args.n_hunters
n_required_hunters = n_hunters
online_lr = input_args.online_lr
prey_type = input_args.prey_type
start_eps = input_args.start_eps
smart_add = input_args.buffer_smart_add
target_lr = input_args.target_lr
temp = input_args.temp
train_thresh = input_args.train_thresh
tracker_logs = input_args.logs
use_lower_model = input_args.use_lower_model
use_higher_model = input_args.use_higher_model
warmup = input_args.warmup

for i in (reversed(range(limits[0], limits[1] + 1)) if use_higher_model else range(limits[0], limits[1] + 1)):
	n_spawn_preys = i
	HUNTER_IDS = [('h%d' % (idx + 1)) for idx in range(n_hunters)]
	PREY_IDS = [('p%d' % (idx + 1)) for idx in range(n_spawn_preys)]
	prey_type = input_args.prey_type
	args = (" --n-agents %d --architecture %s --buffer %d --gamma %f --beta %f --reward %s --iterations %d --batch %d --train-freq %d "
			"--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d --legibility-temp %f "
			"--hunter-ids %s --prey-ids %s --hunter-classes %d --prey-type %s --field-size %d --n-hunters-catch %d --steps-episode %d --catch-reward %f"
			% (n_hunters, ARQUITECTURE, buffer_size, GAMMA, BETA, leg_reward,                                                                           # DQN parameters
			   n_iterations, batch_size, TRAIN_FREQ, TARGET_FREQ, online_lr, target_lr, start_eps, final_eps, eps_decay, eps_type, warmup, temp,        # Train parameters
			   ' '.join(HUNTER_IDS), ' '.join(PREY_IDS), HUNTER_CLASSES, prey_type, field_len, n_required_hunters, max_steps, catch_reward))            # Environment parameters
	args += ((" --dueling" if USE_DUELING else "") + (" --ddqn" if USE_DDQN else "") + (" --render" if USE_RENDER else "") + ("  --gpu" if USE_GPU else "") +
			 (" --cnn" if USE_CNN else "") + (" --tracker" if USE_TRACKER else "") + (" --vdn" if USE_VDN else "") +
			 (" --restart --restart-info %s %s %s" % (RESTART_INFO[0], RESTART_INFO[1], str(RESTART_INFO[2])) if RESTART else "") +
			 (" --debug" if DEBUG else "") + (" --use-opt-vdn" if OPT_VDN else "") + (" --n-leg-agents %d" % N_LEG_AGENTS) + (" --fraction %f" % PRECOMP_FRAC) +
			 (" --models-dir %s" % models_dir if models_dir != '' else "") + (" --data-dir %s" % data_dir if data_dir != '' else "") +
			 (" --logs-dir %s" % logs_dir if logs_dir != '' else "") + (" --use-lower-model" if use_lower_model else "") + (" --use-higher-model" if use_higher_model else "") +
	         (" --buffer-smart-add --buffer-method %s" % add_method if smart_add else "") + (" --tracker-dir %s" % tracker_logs if tracker_logs != '' else "") +
	         (" --train-performance %f" % train_thresh if train_thresh is not None else ""))

	command = "python " + str(src_dir / 'train_pursuit_legible_dqn.py') + args
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

