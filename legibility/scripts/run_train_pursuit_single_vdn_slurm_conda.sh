#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=train_pursuit_single_vdn
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  if [ -z "$CONDA_PREFIX_1" ] ; then
    conda_dir="$CONDA_PREFIX"
  else
    conda_dir="$CONDA_PREFIX_1"
  fi
else
  conda_dir="$CONDA_HOME"
fi

source "$conda_dir"/bin/activate drl_env
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  python "$script_path"/run_train_pursuit_single_vdn_dqn.py --field-len 20 --hunters 2 --catch-reward 4 --prey-type idle --batch-size 64 --buffer-size 5000 --iterations 4000 --episode-steps 1000 --warmup 1000 --limits 4 4 --eps-type linear --eps-decay 0.1 --online-lr 0.00005 --use-higher-curriculum --logs-dir /mnt/scratch-artemis/miguelfaria/logs/pursuit --models-dir /mnt/data-artemis/miguelfaria/deep_rl/models --data-dir /mnt/data-artemis/miguelfaria/deep_rl/data
elif [ "$HOSTNAME" = "hera" ] ; then
  nohup python "$script_path"/run_train_pursuit_single_vdn_dqn.py --field-len 20 --hunters 2 --catch-reward 5 --prey-type idle --batch-size 64 --buffer-size 2500 --iterations 4000 --episode-steps 800 --warmup 800 --limits 1 4 --eps-type log --eps-decay 0.175 --use-lower-curriculum --logs-dir /mnt/data-hera1/miguelfaria/deep_rl/logs/pursuit --models-dir /mnt/data-hera1/miguelfaria/deep_rl/models --data-dir /mnt/data-hera1/miguelfaria/deep_rl/data
else
  python "$script_path"/run_train_pursuit_single_vdn_dqn.py --field-len 20 --hunters 2 --catch-reward 3 --prey-type idle --batch-size 64 --buffer-size 2500 --iterations 2000 --episode-steps 800 --limits 1 1 --eps-type log --start-eps 1.0 --eps-decay 0.175 --use-lower-curriculum
fi

conda deactivate
date
