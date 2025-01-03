#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=mohit_explanation_exec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=gpu-short
#SBATCH --output="job-%x-%j.out"
#SBATCH --partition=a6000

date;hostname;pwd
options=$(getopt -o d:,s:,t:,u:,b: -l mm:,se:,te:,lib:,key:,sgpu:,tgpu:,shost:,thost:,sport:,tport:,temp:,lp:,remote -- "$@")
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  cache_dir="/mnt/scratch-artemis/miguelfaria/llms/checkpoints"
  data_dir="/mnt/data-artemis/miguelfaria/llms/"
else
  cache_dir="./cache"
  data_dir="./data"
fi

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

if [ -z "$api_key" ]; then
    remote_model="token-a1b2c3d4"
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

student_model_url="http://$student_host:$student_port/v1"
teacher_model_url="http://$teacher_host:$teacher_port/v1"

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

cd "$script_path" || exit
cd .. || exit

if [ "$dataset" = "ec_qa" ]; then
  dataset_dir="datasets/ecqa"
  train_file="data_train.csv"
  test_file="data_test.csv"
  val_file="data_val.csv"
elif [ "$dataset" = "gsm8k" ]; then
  dataset_dir="datasets/gsm8k"
  train_file="train.jsonl"
  test_file="test.jsonl"
  val_file=""
else
  dataset_dir="datasets/strategyqa"
  train_file="train.json"
  test_file="test.json"
  val_file="validation.json"
fi

s_name=$(sed 's/-/_/g' <<< "$(sed 's/\//_/g' <<< "$student_model")")
t_name=$(sed 's/-/_/g' <<< "$(sed 's/\//_/g' <<< "$teacher_model")")
out_file=mohit_"$mental_model"_"$t_name"_"$utility"_"$s_name"_"$dataset"_"$(date '+%Y-%m-%d_%H-%M-%S')".out
results_path="$data_dir"/results/mohit_"$mental_model"_"$t_name"_"$utility"_"$s_name"_"$dataset"_"$(date '+%Y-%m-%d_%H-%M-%S')".txt

# export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
# export VLLM_TRACE_FUNCTION=1

gpu_usage=0.7
readarray -d "," -t gpus_avail <<< "$CUDA_VISIBLE_DEVICES"
student_gpus="${gpus_avail[@]:0:$n_student_gpus}"
teacher_gpus="${gpus_avail[@]:$n_student_gpus:$n_teacher_gpus}"
student_gpus="${student_gpus// /,}"
teacher_gpus="${teacher_gpus// /,}"

if [ -z "$remote_model"  ]; then
  python src/mohit_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
                                    --val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" \
                                    --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 4 --n-ic-samples 5 --mm-type "$mental_model" \
                                    --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" --use-explanations \
                                    --use-gold-label --budgets "${budgets[@]}" --llm-lib "$lib" --temperature "$gen_temperature" --n-logprobs "$num_logprobs" > "$out_file"
else
  if [ "$lib" = "vllm" ]; then
    echo "Serving student model using vLLM"
    echo "Student's gpus: $student_gpus"
    CUDA_VISIBLE_DEVICES="$student_gpus" vllm serve "$student_model" --download-dir "$cache_dir" --dtype auto --api-key "$api_key" --gpu-memory-utilization "$gpu_usage" \
                                                    --tensor-parallel-size "$n_student_gpus" --host "$student_host" --port "$student_port" &
    student_id=$!
    # sleep 5m
    echo "Serving teacher model using vLLM"
    echo "Teacher's gpus: $teacher_gpus"
    CUDA_VISIBLE_DEVICES="$teacher_gpus" vllm serve "$teacher_model" --download-dir "$cache_dir" --dtype auto --api-key "$api_key" --gpu-memory-utilization "$gpu_usage" \
                                                    --tensor-parallel-size "$n_teacher_gpus" --host "$teacher_host" --port "$teacher_port" &
    teacher_id=$!
    sleep 5m
  fi
  echo "Launching Mohit's experiment script"
  python src/mohit_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
                                    --val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" \
                                    --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 4 --n-ic-samples 5 --mm-type "$mental_model" \
                                    --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" --use-explanations \
                                    --use-gold-label --budgets "${budgets[@]}" --llm-lib "$lib" --remote --student-model-url "$student_model_url" --teacher-model-url "$teacher_model_url" --api-key "$api_key" \
                                    --temperature "$gen_temperature" --n-logprobs "$num_logprobs" > "$out_file"
  if [ "$lib" = "vllm" ]; then
    kill -9 "$student_id" "$teacher_id"
  fi
fi

conda deactivate
date
