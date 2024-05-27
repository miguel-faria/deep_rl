#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_lb_foraging_vdn_legible
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=shard:10
#SBATCH --time=336:00:00
#SBATCH --mem-per-cpu=4000
date;hostname;pwd

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  script_path=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/cuda/bin:$PATH"
source "$HOME"/miniconda3/bin/activate deep_rl_env

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
python "$script_path"/run_train_lb_vdn_legible_dqn.py --field-len 8 --iterations 600 --limits 1 4

source "$HOME"/miniconda3/bin/deactivate
date
