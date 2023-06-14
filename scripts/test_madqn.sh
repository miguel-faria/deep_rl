#!/bin/bash

date

script_dir="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
script_dir_parent="$( cd -- "$script_dir" >/dev/null 2>&1 ; cd ..; pwd -P )"

# DQN params
N_AGENTS=2
N_LAYERS=3
BUFFER=10000
GAMMA=0.95
LAYERS=(128, 128, 84)
AGENTS=("agent_1", "agent_2")

# Train Params
N_STEPS=100000
BATCH_SIZE=128
TRAIN_FREQ=10
TARGET_FREQ=500
ALPHA=0.00025
TAU=0.0000025
INIT_EPS=1.0
FINAL_EPS=0.01
EPS_DECAY=0.95
EPS_TYPE="linear"
WARMUP_STEPS=10000

python_args="--nagents $N_AGENTS --nlayers $N_LAYERS --buffer $BUFFER --gamma $GAMMA --layer-sizes ${LAYERS[@]} --agent-ids ${AGENTS[@]} --gpu\
			 --steps $N_STEPS --batch $BATCH_SIZE --train-freq $TRAIN_FREQ --target-freq $TARGET_FREQ --alpha $ALPHA --tau $TAU --init-eps $INIT_EPS\
		 	 --final-eps $FINAL_EPS --eps-decay $EPS_DECAY --eps-type $EPS_TYPE --warmup-steps $WARMUP_STEPS"

python "$script_dir_parent/src/test_madqn.py" $python_args