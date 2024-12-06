#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=test_lb_legible_collaboration
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=4G
#SBATCH --qos=gpu-medium
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
  python "$script_path"/run_test_lb_legible_collaboration.py --tests 250 --mode 2 --logs-dir /mnt/scratch-artemis/miguelfaria/logs/lb-foraging --models-dir /mnt/data-artemis/miguelfaria/deep_rl/models --data-dir /mnt/data-artemis/miguelfaria/deep_rl/data
else
  python "$script_path"/run_test_lb_legible_collaboration.py --tests 250 --mode 2
fi

conda deactivate
date
