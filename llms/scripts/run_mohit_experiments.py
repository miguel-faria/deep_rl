#! /usr/bin/env python
import shlex
import subprocess
import time

from pathlib import Path


src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
USE_SHELL = False

TASK = 'strategy_qa'
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

MAX_TOKENS = 100
N_BEAMS = 4
N_SAMPLES = 5

USE_EXPLANATION = True
USE_DECEPTION = False
USE_GOLD_LABEL = True

args = (" --data-dir %s --cache-dir %s --train-filename %s --test-filename %s --val-filename %s --results-path %s --task %s --student-model %s --teacher-model %s --max-new-tokens %d"
		" --n-beams %d --n-ic-samples %d --mm-type %s --intervene-behaviour %s --intervention-utility %s --teacher-explanation-type %s --student-explanation-type %s"
		% (DATASET_DIR, CACHE_DIR, TRAIN_FILE, TEST_FILE, VALIDATION_FILE, RESULTS_FILE, TASK, STUDENT_MODEL, TEACHER_MODEL, MAX_TOKENS, N_BEAMS, N_SAMPLES,
		   MM_TYPE, INTERVENE_BEHAVIOUR, INTERVENTION_UTILITY, TEACHER_EXPLANATION, STUDENT_EXPLANATION))
args += ((' --use-explanations' if USE_EXPLANATION else '') + (' --deceive' if USE_DECEPTION else '') + (' --use-gold-label' if USE_GOLD_LABEL else ''))

commamd = "python " + str(src_dir / 'mohit_mm_experiments.py') + args
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