#!/bin/bash

date;hostname;pwd
options=$(getopt -o d:,s:,t:,u:,b: -l mm:,se:,te:,lib:,key:,sgpu:,tgpu:,shost:,thost:,sport:,tport:,temp:,lp:,usage:,remote -- "$@")

eval set -- "$options"
budgets=()

while [ $# -gt 0 ]
do
  case $1 in
    -d) dataset=${2}; shift ;;
    -s) student_model=${2}; shift ;;
    -t) teacher_model=${2}; shift ;;
    -u) utility=${2}; shift ;;
    -b) budgets+=("${2}"); shift ;;
    --lib) lib=${2}; shift ;;
    --mm) mental_model=${2}; shift ;;
    --se) student_expl=${2}; shift ;;
    --te) teacher_expl=${2}; shift ;;
    --remote) remote_model=1 ;;
    --key) api_key=${2}; shift ;;
    --sgpu) n_student_gpus=${2}; shift ;;
    --tgpu) n_teacher_gpus=${2}; shift ;;
    --shost) student_host=${2}; shift ;;
    --thost) teacher_host=${2}; shift ;;
    --sport) student_port=${2}; shift ;;
    --tport) teacher_host=${2}; shift ;;
    --temp) gen_temperature=${2}; shift ;;
    --lp) num_logprobs=${2}; shift ;;
    --usage) gpu_usage=${2}; shift ;;
    (--) shift; break ;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1 ;;
    (*) break ;;
    esac
    shift
done

if [ ${#budgets[@]} -eq 0 ]; then
  budgets=("0" "0.2" "0.4" "0.6" "0.8" "1.0")
fi

if [ -z "$dataset" ]; then
  dataset="strategy_qa"
fi

if [ -z "$lib" ]; then
  lib="vllm"
fi

if [ -z "$student_model" ]; then
  student_model="google/flan-t5-large"
fi

if [ -z "$teacher_model" ]; then
  teacher_model="google/flan-t5-xl"
fi

if [ -z "$mental_model" ]; then
  mental_model="mm_both"
fi

if [ -z "$utility" ]; then
  utility="mm_both"
fi

if [ -z "$student_expl" ]; then
  student_expl="cot"
fi

if [ -z "$teacher_expl" ]; then
  teacher_expl="useful_teacher"
fi

if [ -z "$remote_model" ]; then
    remote_model=0
fi

if [ -z "$api_key" ]; then
    api_key="token-a1b2c3d4"
fi

if [ -z "$n_student_gpus" ]; then
    n_student_gpus="1"
fi

if [ -z "$n_teacher_gpus" ]; then
    n_teacher_gpus="2"
fi

if [ -z "$student_host" ]; then
    student_host="localhost"
fi

if [ -z "$teacher_host" ]; then
    teacher_host="localhost"
fi

if [ -z "$student_port" ]; then
    student_port=15050
fi

if [ -z "$teacher_port" ]; then
    teacher_port=15051
fi

if [ -z "$gen_temperature" ]; then
    gen_temperature=0.0
fi

if [ -z "$num_logprobs" ]; then
    num_logprobs=5
fi

if [ -z "$gpu_usage" ]; then
    gpu_usage=0.7
fi

script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"

if [ -z "$remote_model" ]; then
  sbatch "$script_path"/launch_mohit_experiments_slurm.sh -d "$dataset" -s "$student_model" -t "$teacher_model" -u "$utility" --mm "$mental_model" --se "$student_expl" --te "$teacher_expl" --lib "$lib" \
                                                          --key "$api_key" --sgpu "$n_student_gpus" --tgpu "$n_teacher_gpus" --shost "$student_host" --thost "$teacher_host" --sport "$student_port" \
                                                          --tport "$teacher_port" --temp "$gen_temperature" --lp "$num_logprobs" --usage "$gpu_usage"
else
  sbatch "$script_path"/vllm_serve_teacher_model_slurm.sh -t "$teacher_model" -u "$gpu_usage" --key "$api_key" --gpu "$n_teacher_gpus" --host "$teacher_host" --port "$teacher_port" --temp "$gen_temperature"
  sbatch "$script_path"/launch_mohit_experiments_slurm.sh -d "$dataset" -s "$student_model" -t "$teacher_model" -u "$utility" --mm "$mental_model" --se "$student_expl" --te "$teacher_expl" --lib "$lib" \
                                                          --key "$api_key" --sgpu "$n_student_gpus" --tgpu "$n_teacher_gpus" --shost "$student_host" --thost "$teacher_host" --sport "$student_port" \
                                                          --tport "$teacher_port" --temp "$gen_temperature" --lp "$num_logprobs" --usage "$gpu_usage" --remote
fi

date
