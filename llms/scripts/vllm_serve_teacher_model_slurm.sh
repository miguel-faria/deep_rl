#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=vllm_serve_teacher
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=gpu-short
#SBATCH --output="job-%x-%j.out"
#SBATCH --partition=a6000

date;hostname;pwd
options=$(getopt -o t:,u: -l key:,gpu:,host:,port:,temp: -- "$@")
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  cache_dir="/mnt/scratch-artemis/miguelfaria/llms/checkpoints"
else
  cache_dir="./cache"
fi

eval set -- "$options"

while [ $# -gt 0 ]
do
  case $1 in
    -t) teacher_model=${2}; shift ;;
    -u) gpu_usage=${2}; shift ;;
    --key) api_key=${2}; shift ;;
    --gpu) n_teacher_gpus=${2}; shift ;;
    --host) teacher_host=${2}; shift ;;
    --port) teacher_port=${2}; shift ;;
    --temp) gen_temperature=${2}; shift ;;
    (--) shift; break ;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1 ;;
    (*) break ;;
    esac
    shift
done

if [ -z "$teacher_model" ]; then
  teacher_model="google/flan-t5-xl"
fi

if [ -z "$api_key" ]; then
    api_key="token-a1b2c3d4"
fi

if [ -z "$n_teacher_gpus" ]; then
    n_teacher_gpus="2"
fi

if [ -z "$teacher_host" ]; then
    teacher_host="localhost"
fi

if [ -z "$teacher_port" ]; then
    teacher_port=15051
fi

if [ -z "$gen_temperature" ]; then
    gen_temperature=0.0
fi

if [ -z "$gpu_usage" ]; then
    gpu_usage=0.7
fi

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  IFS=' '
  read -ra newarr <<< "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')"
  script_path=$(dirname "${newarr[0]}")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

# module load python cuda
export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/cuda/bin:$PATH"
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  if [ -z "$CONDA_PREFIX_1" ] ; then
    conda_dir="$CONDA_PREFIX"
  else
    conda_dir="$CONDA_PREFIX_1"
  fi
else
  conda_dir="$CONDA_HOME"
fi

source "$conda_dir"/bin/activate llm_env

echo "Serving teacher model using vLLM"
vllm serve "$teacher_model" --download-dir "$cache_dir" --dtype auto --api-key "$api_key" --gpu-memory-utilization "$gpu_usage" \
                            --tensor-parallel-size "$n_teacher_gpus" --host "$teacher_host" --port "$teacher_port"

conda deactivate
date
