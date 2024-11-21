#!/bin/bash

date;hostname;pwd
options=$(getopt -o n:,m:,s:,j: -l food:,spawn_foods:,field: -- "$@")
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  logs_dir="/mnt/scratch-artemis/miguelfaria/logs/lb-foraging"
  data_dir="/mnt/data-artemis/miguelfaria/deep_rl/data"
  models_dir="/mnt/data-artemis/miguelfaria/deep_rl/models"
else
  logs_dir="./logs"
  data_dir="./data"
  models_dir="./models"
fi

eval set -- "$options"

while [ $# -gt 0 ]
do
  case $1 in
    -n) n_tests=${2}; shift ;;
    -m) test_mode=${2}; shift ;;
    -s) start_run=${2}; shift ;;
    -j) tests_job=${2}; shift ;;
    --foods) max_foods=${2}; shift ;;
    --spawn_foods) max_spawn_foods=${2}; shift ;;
    --field) field_len=${2}; shift ;;
    (--) shift; break ;;
    (-*) echo "$0: error - unrecognized option $1" 1>&2; exit 1 ;;
    (*) break ;;
    esac
    shift
done

if [ -z "$n_tests" ]; then
  n_tests=250
fi

if [ -z "$test_mode" ]; then
  test_mode=2
fi

if [ -z "$start_run" ]; then
  start_run=0
fi

if [ -z "$tests_job" ]; then
  tests_job=10
fi

if [ -z "$max_foods" ]; then
  max_foods=8
fi

if [ -z "$max_spawn_foods" ]; then
  max_spawn_foods=6
fi

if [ -z "$field_len" ]; then
  field_len=8
fi

n_jobs=$(( (n_tests + tests_job - 1) / tests_job ))

if [ -n "${SLURM_JOB_ID:-}" ] ; then
  IFS=' '
  read -ra newarr <<< "$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')"
  script_path=$(dirname "${newarr[0]}")
else
  script_path="$( cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )"
fi

#module load python cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3
source "$HOME"/miniconda3/bin/activate drl_env
if [ "$HOSTNAME" = "artemis" ] || [ "$HOSTNAME" = "poseidon" ] ; then
  for ((job=1; job<=num_jobs; job++)); do
    start_test=$(( (job - 1) * tests_per_job + 1 ))
    end_test=$(( job * tests_per_job ))

    # Adjust the end test for the last job if it exceeds the total tests
    if [ $end_test -gt $total_tests ]; then
      end_test=$total_tests
    fi

    # Generate the sbatch script for this job
    if [ "$job" -gt 1 ] ; then
      job_id=$(sbatch --parsable << EOF
        #!/bin/bash
        #SBATCH --mail-type=BEGIN,END,FAIL
        #SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
        #SBATCH --job-name=test_lb_legible_collaboration_"$job"
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=1
        #SBATCH --tasks-per-node=1
        #SBATCH --gres=gpu:1
        #SBATCH --time=04:00:00
        #SBATCH --mem=4G
        #SBATCH --qos=gpu-short
        #SBATCH --output=job-%x-%j_"$job".out
        #SBATCH --partition=a6000

        python "$script_path"/run_test_lb_legible_collaboration.py --tests "$end_test" --start-run "$start_test" --mode "$test_mode" --field-len "$field_len" --max-foods "$max_foods" --spawn-foods "$max_spawn_foods" --logs-dir "$logs_dir" --models-dir "$models_dir" --data-dir "$data_dir"
        EOF
      )
    else
      job_id=$(sbatch --parsable << EOF
        #!/bin/bash
        #SBATCH --mail-type=BEGIN,END,FAIL
        #SBATCH --mail-user=miguel.faria@tecnico.ulisboa.pt
        #SBATCH --job-name=test_lb_legible_collaboration_"$job"
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=1
        #SBATCH --tasks-per-node=1
        #SBATCH --gres=gpu:1
        #SBATCH --time=04:00:00
        #SBATCH --mem=4G
        #SBATCH --qos=gpu-short
        #SBATCH --output=job-%x-%j_"$job".out
        #SBATCH --partition=a6000
        #SBATCH --dependency=afterok:"$job_id"

        python "$script_path"/run_test_lb_legible_collaboration.py --tests "$end_test" --start-run "$start_test" --mode "$test_mode" --field-len "$field_len" --max-foods "$max_foods" --spawn-foods "$max_spawn_foods" --logs-dir "$logs_dir" --models-dir "$models_dir" --data-dir "$data_dir"
        EOF
      )
    fi
    echo "Job ID: "$job_id""
  done
else
  python "$script_path"/run_test_lb_legible_collaboration.py --tests "$n_tests" --mode "$test_mode"
fi

source "$HOME"/miniconda3/bin/deactivate
date
