#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
#SBATCH --job-name=mohit_explanation_exec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --qos=gpu-medium
#SBATCH --output="job-%x-%j.out"

date;hostname;pwd
options=$(getopt -o d:,s:,t:,mm:,u:,se:,te:,ss:,it: -- "$@")
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  cache_dir="/mnt/scratch-artemis/miguelfaria/llms/checkpoints"
  data_dir="/mnt/data-artemis/miguelfaria/llms/"
else
  cache_dir="./cache"
  data_dir="./data"
fi

eval set -- "$options"

while [ $# -gt 0 ]
do
  case $1 in
    -d) dataset=${2}; shift ;;
    -s) student_model=${2}; shift ;;
    -t) teacher_model=${2}; shift ;;
    -mm) mental_model=${2}; shift ;;
    -u) utility=${2}; shift ;;
    -se) student_expl=${2}; shift ;;
    -te) teacher_expl=${2}; shift ;;
    -ss) student_samples=${2}: shift ;;
    -it) intervention_thresh=${2}: shift ;;
    (--) shift; break ;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1 ;;
    (*) break ;;
    esac
    shift
done

if [ -z "$dataset" ]; then
  dataset="strategy_qa"
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

if [ -z "$intervention_thresh" ]; then
  intervention_thresh=0.5
fi

if [ -z "$student_samples" ]; then
  student_samples=10
fi

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  script_path=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | head -n 1)")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

results_path="$data_dir"/results/mohit_"$mental_model"_"$utility".txt

export LD_LIBRARY_PATH="/opt/cuda/lib64:$LD_LIBRARY_PATH"
export PATH="/opt/cuda/bin:$PATH"
source "$HOME"/miniconda3/bin/activate tom_env

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



python src/mohit_mm_experiments.py --data-dir "$data_dir"/"$dataset_dir" --cache-dir "$cache_dir" --train-filename "$train_file" --test-filename "$test_file" \
--val-filename "$val_file" --results-path "$results_path" --task "$dataset" --student-model "$student_model" --teacher-model "$teacher_model" --max-new-tokens 100 --n-beams 4 \
--n-ic-samples 5 --mm-type "$mental_model" --intervention-utility "$utility" --teacher-explanation-type "$teacher_expl" --student-explanation-type "$student_expl" \
--use-explanations --use-gold-label --intervention-threshold "$intervention_thresh" --max-student-samples "$student_samples" > mohit_"$mental_model"_"$utility".out

source "$HOME"/miniconda3/bin/deactivate
date
