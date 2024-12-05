#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_lb_foraging_vdn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00
#SBATCH --mem=4G
#SBATCH --qos=gpu-long
#SBATCH --output="job-%x-%j.out"
#SBATCH --partition=a6000

date;hostname;pwd

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  script_path=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

#export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
#export PATH="/opt/cuda/bin:$PATH"

#module load python cuda
source ~/.bashrc
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  source "$CONDA_PREFIX"/bin/activate drl_env
else
  source "$CONDA_HOME"/bin/activate drl_env
fi
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  python "$script_path"/run_train_lb_vdn_single_dqn.py --field-len 20 --batch-size 64 --buffer-size 2500 --iterations 5000 --episode-steps 1000 --limits 1 7 --eps-type log --start-epsh 0.5 --eps-decay 0.09 --use-higher-curriculum --train-thresh 0.9 --version v2 --logs-dir /mnt/scratch-artemis/miguelfaria/logs/lb-foraging --models-dir /mnt/data-artemis/miguelfaria/deep_rl/models --data-dir /mnt/data-artemis/miguelfaria/deep_rl/data
else
  python "$script_path"/run_train_lb_vdn_single_dqn.py --field-len 15 --batch-size 64 --buffer-size 5000 --iterations 5000 --episode-steps 600 --limits 1 8 --eps-type log --eps-decay 0.07 --use-higher-curriculum --train-thresh 0.9 --version v2
fi

source "$HOME"/miniconda3/bin/deactivate
date
