#!/bin/bash
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gres:shard=NR_GB_GPU
#SBATCH --gres=gpu_mem:20000
#SBATCH --time=144:00:00
#SBATCH --mem-per-cpu=1000

python $HOME/Documents/Projects/deep_rl/tests/run_train_lb_dqn.py