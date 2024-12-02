#! /usr/bin/env python
import shlex
import subprocess
import time
import argparse

from pathlib import Path


src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
USE_SHELL = False

TASK = 'strategy_qa'
LLM_LIB = 'vllm'
DATASET_DIR = data_dir / 'datasets' / 'strategyqa'
CACHE_DIR = data_dir.parent.absolute() / 'cache'
TRAIN_FILE = 'train.json'
TEST_FILE = 'test.json'
VALIDATION_FILE = 'validation.json'
RESULTS_FILE = data_dir / 'results' / 'results_mm_both.txt'

STUDENT_MODEL = 'google/flan-t5-large'
TEACHER_MODEL = 'google/flan-t5-xl'
MM_TYPE = 'mm_both'
TEACHER_EXPLANATION = 'useful_teacher'
STUDENT_EXPLANATION = 'cot'
INTERVENE_BEHAVIOUR = 'teacher'
INTERVENTION_UTILITY = 'mm_both'
INTERVENTION_THRESH = 0.1
MAX_STUDENT_SAMPLES = 10

BUDGETS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
MAX_TOKENS = 100
N_BEAMS = 1
N_SAMPLES = 5

USE_EXPLANATIONS = True
USE_DECEPTION = False
USE_GOLD_LABEL = True

parser = argparse.ArgumentParser(description='Interactive teacher experiments')
parser.add_argument('--llm-lib', dest='llm_lib', default=LLM_LIB, type=str, choices=['hf', 'vllm'], help='LLM transformer lib to use, either HuggingFace (hf) or vLLM (vllm)')
parser.add_argument('--budgets', dest='budgets', default=BUDGETS, type=float, nargs='+',
                    help='Interaction budgets to test the teaching. Default: [0, 0.2, 0.4, 0.6, 0.8, 1.0]')
parser.add_argument('--cache-dir', dest='cache_dir', default=CACHE_DIR, type=str, help='Path to the cache directory, where downladed models are stored')
parser.add_argument('--data-dir', dest='data_dir', default=DATASET_DIR, type=str, help='Path to the directory with the datasets')
parser.add_argument('--intervention-threshold', dest='intervention_threshold', default=INTERVENTION_THRESH, type=float,
					help='Threshold for intervention utility, above which the mental model gives an explanation')
parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=MAX_TOKENS, type=int, help='Maximum number of new tokens when generating answers')
parser.add_argument('--max-student-samples', dest='max_student_samples', default=MAX_STUDENT_SAMPLES, type=int, help='Maximum number of students to sample for mental model context')
parser.add_argument('--mm-type', dest='mm_type', default=MM_TYPE, type=str, help='Mental model intervention strategy')
parser.add_argument('--n-beams', dest='n_beams', default=N_BEAMS, type=int, help='Number of beams to use in answer generation beam search')
parser.add_argument('--n-ic-samples', dest='n_ics', default=N_SAMPLES, type=int, help='Number of in-context samples to use for context in the student answers')
parser.add_argument('--results-path', dest='results_path', default=RESULTS_FILE, type=str, help='Path to the results file')
parser.add_argument('--student-explanation-type', dest='student_expl_type', default=STUDENT_EXPLANATION, type=str, help='Student model explanation type')
parser.add_argument('--student-model', dest='student_model', default=STUDENT_MODEL, type=str,
					help='Local or hugging face path to use for the student model')
parser.add_argument('--task', dest='task', default=TASK, choices=['strategy_qa', 'ec_qa', 'gsm8k'], type=str, help='Dataset task to run')
parser.add_argument('--teacher-explanation-type', dest='teacher_expl_type', default=TEACHER_EXPLANATION, type=str, help='Teacher model explanation type')
parser.add_argument('--teacher-model', dest='teacher_model', default=TEACHER_MODEL, type=str,
					help='Local or hugging face path to use for the teacher model')

input_args = parser.parse_args()

budgets = input_args.budgets
cache_dir = input_args.cache_dir
dataset_dir = input_args.data_dir
llm_lib = input_args.llm_lib
intervention_threshold = input_args.intervention_threshold
max_new_tokens = input_args.max_new_tokens
max_student_samples = input_args.max_student_samples
mm_type = input_args.mm_type
n_beams = input_args.n_beams
n_ics = input_args.n_ics
results_file = input_args.results_path
student_expl_type = input_args.student_expl_type
student_model = input_args.student_model
task = input_args.task
teacher_expl_type = input_args.teacher_expl_type
teacher_model = input_args.teacher_model


args = (" --data-dir %s --cache-dir %s --train-filename %s --test-filename %s --val-filename %s --results-path %s --task %s --student-model %s --teacher-model %s"
        " --max-new-tokens %d --n-beams %d --n-ic-samples %d --max-student-samples %d --mm-type %s --intervene-behaviour %s --intervention-utility %s --teacher-explanation-type %s"
        " --student-explanation-type %s --intervention-threshold %f --llm-lib %s"
        % (dataset_dir, cache_dir, TRAIN_FILE, TEST_FILE, VALIDATION_FILE, results_file, task, student_model, teacher_model, max_new_tokens, n_beams, n_ics,
           max_student_samples, mm_type, INTERVENE_BEHAVIOUR, INTERVENTION_UTILITY, teacher_expl_type, student_expl_type, intervention_threshold, llm_lib))
args += ((' --use-explanations' if USE_EXPLANATIONS else '') + (' --deceive' if USE_DECEPTION else '') + (' --use-gold-label' if USE_GOLD_LABEL else ''))

commamd = "python " + str(src_dir / 'interactive_mm_experiments.py') + args
if not USE_SHELL:
	commamd = shlex.split(commamd)

print(commamd)
start_time = time.time()
try:
	subprocess.run(commamd, shell=USE_SHELL, check=True)

except subprocess.CalledProcessError as e:
	print(e.output)

except KeyboardInterrupt as ki:
	print('Caught keyboard interrupt by user: %s Exiting....' % ki)

except Exception as e:
	print('Caught general exception: %s' % e)

wall_time = time.time() - start_time
print('Finished training, took %.3f seconds' % wall_time)