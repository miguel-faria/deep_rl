#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=test_lb_single_dqn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=shard:10
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=1000
date;hostname;pwd

source $HOME/python_envs/deep_rl_env/bin/activate
python $HOME/Documents/Projects/deep_rl/scripts/run_test_lb_single_dqn.py

deactivate
date
