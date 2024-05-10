#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_lb_foraging_multi_env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=shard:10
#SBATCH --time=336:00:00
#SBATCH --mem-per-cpu=1000
date;hostname;pwd

export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/cuda/bin:$PATH"
source "$HOME"/miniconda3/bin/activate deep_rl_env

python $HOME/Documents/Projects/deep_rl/scripts/run_train_lb_single_multi_env_vdn_dqn.py --field-len 8 --episode-steps 400 --iterations 400 --limits 1 5 --tags multi_env

deactivate
date
