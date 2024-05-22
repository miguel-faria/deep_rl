#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_lb_foraging_vdn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=shard:10
#SBATCH --time=336:00:00
#SBATCH --mem-per-cpu=1000
date;hostname;pwd

source $HOME/python_envs/deep_rl_env/bin/activate
export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/cuda/bin:$PATH"

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
python $HOME/Documents/Projects/deep_rl/scripts/run_train_lb_single_vdn_dqn.py --field-len 20

deactivate
date
