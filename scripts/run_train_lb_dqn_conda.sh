#!/bin/bash

date;hostname;pwd

# conda activate drl_env
python $HOME/Documents/Projects/deep_rl/scripts/run_train_lb_dqn.py

# conda deactivate
date

