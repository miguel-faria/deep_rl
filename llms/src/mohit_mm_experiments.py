#! /usr/bin/env python

import argparse
import pandas as pd
import torch

from reputation_learning.dataset_tasks_utils import ECQA, StrategyQA, GSM8k
from reputation_learning.model import UnidentifiedTaskError, UnidentifiedExplanationError
from reputation_learning.teacher_model import TeacherModel
from reputation_learning.student_model import StudentModel
from reputation_learning.teacher_mental_model import TeacherMentalModel
from pathlib import Path
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, List, Optional
from numpy.random import default_rng


RNG_SEED = 25092024


def load_models(rng_seed: int, train_data: pd.DataFrame, num_samples: int, student_model_path: str, teacher_model_path: str, task: str, use_explanations: bool,
                expl_type: str, intervene_strat: str, intervene_behaviour: str, max_tokens: int, num_beams: int, cache_dir: Path) -> Tuple[StudentModel, Optional[TeacherModel], Optional[TeacherMentalModel]]:

	rng_gen = default_rng(rng_seed)
	train_idxs = rng_gen.choice(train_data.shape[0], num_samples, replace=False)
	student_samples = [train_data.iloc[idx] for idx in train_idxs]

	student_tokenizer = AutoTokenizer.from_pretrained(student_model_path, cache_dir=cache_dir, use_fast=False)
	teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, cache_dir=cache_dir, use_fast=False) if teacher_model_path != 'human' else None

	if "llama" in student_model_path:
		student_gen_model = AutoModelForCausalLM.from_pretrained(student_model_path, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16)
	else:
		student_gen_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_path, device_map="auto", cache_dir=cache_dir)

	student_model = StudentModel(student_model_path, student_samples, student_gen_model, student_tokenizer, expl_type, task, max_tokens, num_beams, use_explanations)

	if use_explanations:
		if expl_type.find('human') != -1:
			teacher_model = TeacherModel(teacher_model_path)

		else:
			

	else:
		return student_model, None, None


def main():
	parser = argparse.ArgumentParser(description='Machine teaching with Theory of Mind based mental models experiments from Mohit Bensal')
	parser.add_argument('--data-dir', dest='data_dir', default='', type=str, help='Path to the directory with the datasets')
	parser.add_argument('--train-filename', dest='train_filename', default='', type=str, help='Filename of the training data')
	parser.add_argument('--test-filename', dest='test_filename', default='', type=str, help='Filename of the testing data')
	parser.add_argument('--val-filename', dest='val_filename', default='', type=str, help='Filename of the validation data')
	parser.add_argument('--task', dest='task', default='strategy_qa', choices=['strategy_qa', 'ec_qa', 'gsm8k'], type=str, help='Dataset task to run')
	parser.add_argument('--student-model', dest='student_model', default='google/flan-t5-large', type=str,
	                    help='Local or hugging face path to use for the student model')
	parser.add_argument('--teacher-model', dest='teacher_model', default='google/flan-t5-xl', type=str,
	                    help='Local or hugging face path to use for the teacher model')

	parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=100, type=int, help='Maximum number of new tokens when generating answers')
	parser.add_argument('--cache-dir', dest='cache_dir', default='', type=str, help='Path to the cache directory, where downladed models are stored')
	parser.add_argument('--n-beams', dest='n_beams', default=1, type=int, help='Number of beams to use in answer generation beam search')
	parser.add_argument('--n-ic-samples', dest='n_ics', default=4, type=int, help='Number of in-context samples to use for context in the student answers')
	parser.add_argument('--use-explanations', dest='use_explanations', action='store_true',
	                    help='Flag denoting whether student is given explanations to help understanding the problem')
	parser.add_argument('--intervene-strat', dest='intervene_strat', default='mm_both', type=str, help='Mental model intervention strategy')
	parser.add_argument('--intervene-behaviour', dest='intervene_behaviour', default='teacher', type=str, help='Teacher intervention behaviour')
	parser.add_argument('--explanation-type', dest='expl_type', default='blind_teacher_CoT', type=str, help='Teacher explanation type')
	parser.add_argument('--deceive', dest='deceive', action='store_true', help='Flag denoting whether teacher gives deceiving explanations')

	parser.add_argument('--use-gold-label', dest='use_gold_label', action='store_true',
	                    help='Flag denoting whether teacher uses the expected answers instead of its own')
	parser.add_argument('--results-path', dest='results_path', default='', type=str, help='Path to the results file')

	args = parser.parse_args()

	if args.task == "strategy_qa":
		task_dataset = StrategyQA(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	elif args.task == "ec_qa":
		task_dataset = ECQA(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	elif args.task == "gsm8k":
		task_dataset = GSM8k(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	else:
		raise UnidentifiedTaskError('Task %s is not defined' % args.task)

	load_models(RNG_SEED, task_dataset.get_train_samples(), args.n_ics, args.student_model, args.teacher_model, args.task, args.use_explanations, args.expl_type,
	            args.intervene_strat, args.intervene_behaviour, args.max_new_tokens, args.cache_dir)


if __name__ == '__main__':
	main()
