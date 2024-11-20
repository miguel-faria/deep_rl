#! /usr/bin/env python

import argparse
import pandas as pd
import torch
import os

from utilities.dataset_tasks_utils import ECQA, StrategyQA, GSM8k
from machine_teaching.models.hf.model_hf import UnidentifiedTaskError
from machine_teaching.models.hf.teacher_model_hf import TeacherModel
from machine_teaching.models.hf.student_model_hf import StudentModel
from machine_teaching.reputation_based_student import ReuptationBasedStudent
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, List, Optional, Dict
from numpy.random import default_rng, Generator

RNG_SEED = 25092024


def get_teacher_model_samples(rng_gen: Generator, train_data: pd.DataFrame, student_samples: List[pd.Series], teacher_expl_type: str, num_samples: int,
							  student_model: StudentModel, teacher_model: TeacherModel = None) -> List[Dict]:
	
	teacher_samples = []
	
	if teacher_expl_type.find('blind') != -1:
		teacher_samples = student_samples
	
	elif teacher_expl_type.find('useful') != -1:
		shuffle_train = train_data.sample(frac=1, random_state=rng_gen).reset_index(drop=True)
		idx = 0
		while len(teacher_samples) < num_samples:
			
			sample = shuffle_train.iloc[idx].to_dict()
			
			student_prediction_no_intervene, _ = student_model.predict(sample, '', False)  # get student prediction without teacher intervention
			
			teacher_expl = sample['explanation'] if teacher_model is None else teacher_model.predict(sample)[0]
			student_prediction_intervene, _ = student_model.predict(sample, teacher_expl, True)  # get student prediction with teacher intervention
			
			if student_prediction_intervene == sample['answer'] and student_prediction_no_intervene != student_prediction_intervene:  # add sample if the intervention made student right
				teacher_samples.append(sample)
			
			idx += 1
	
	else:
		samples_idxs = rng_gen.choice(train_data.shape[0], num_samples, replace=False)
		teacher_samples = [train_data.iloc[x].to_dict() for x in samples_idxs]
	
	return teacher_samples


def load_models(rng_seed: int, train_data: pd.DataFrame, num_samples: int, student_model_path: str, teacher_model_paths: List[str], task: str, use_explanations: bool, student_expl_type: str,
				teacher_expl_type: str, reputation_type: str, intervention_utility: str, max_tokens: int, num_beams: int, max_rep: int, cache_dir: Path) -> Tuple[ReuptationBasedStudent, Optional[List[TeacherModel]]]:
	
	rng_gen = default_rng(rng_seed)
	
	print('Setting up the Student Model')
	train_idxs = rng_gen.choice(train_data.shape[0], num_samples, replace=False)
	student_samples = [train_data.iloc[idx].to_dict() for idx in train_idxs]
	
	student_tokenizer = AutoTokenizer.from_pretrained(student_model_path, cache_dir=cache_dir, use_fast=False)
	
	if "llama" in student_model_path:
		student_gen_model = AutoModelForCausalLM.from_pretrained(student_model_path, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16)
	else:
		student_gen_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_path, device_map="auto", cache_dir=cache_dir)
	
	tmp_student_model = StudentModel(student_model_path, student_samples, student_gen_model, student_tokenizer, student_expl_type, task, max_tokens, num_beams, use_explanations)
	
	if use_explanations:
		print('Setting up the Teacher Models')
		if student_expl_type.find('human') != -1:
			teacher_models = []
			for teacher_path in teacher_model_paths:
				teacher_model = TeacherModel(teacher_path)
				teacher_models.append(teacher_model)
		
		else:
			print('Getting teacher samples')
			teacher_samples = get_teacher_model_samples(rng_gen, train_data, student_samples, teacher_expl_type, num_samples, tmp_student_model)
			
			print('Creating Teacher Models')
			teacher_models = []
			for teacher_path in teacher_model_paths:
				if "llama" in teacher_model_paths:
					teacher_gen_model = AutoModelForCausalLM.from_pretrained(teacher_model_paths, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16)
				else:
					teacher_gen_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_paths, device_map="auto", cache_dir=cache_dir)
				teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, cache_dir=cache_dir, use_fast=False) if teacher_model_paths != 'human' else None
				
				teacher_model = TeacherModel(teacher_path, teacher_samples, teacher_gen_model, teacher_tokenizer, teacher_expl_type, task, max_tokens, num_beams, use_explanations)
				teacher_models.append(teacher_model)
			
			student_model = ReuptationBasedStudent(student_model_path, student_samples, student_gen_model, student_tokenizer, teacher_models, student_expl_type, task, max_tokens, num_beams,
												   use_explanations, intervention_utility, reputation_type, max_rep)
			
			return student_model, teacher_models
	
	else:
		student_model = ReuptationBasedStudent(student_model_path, student_samples, student_gen_model, student_tokenizer, [], student_expl_type, task, max_tokens, num_beams,
											   use_explanations)
		
		return student_model, None


def compute_accuracy(labels, predictions):
	correct = 0
	for (label, prediction) in zip(labels, predictions):
		if label == prediction:
			correct += 1
	
	return correct / len(labels)


def main( ):
	parser = argparse.ArgumentParser(description='Machine teaching with Theory of Mind based mental models experiments from Mohit Bensal')
	
	parser.add_argument('--data-dir', dest='data_dir', default='', type=str, help='Path to the directory with the datasets')
	parser.add_argument('--cache-dir', dest='cache_dir', default='', type=str, help='Path to the cache directory, where downladed models are stored')
	parser.add_argument('--train-filename', dest='train_filename', default='', type=str, help='Filename of the training data')
	parser.add_argument('--test-filename', dest='test_filename', default='', type=str, help='Filename of the testing data')
	parser.add_argument('--val-filename', dest='val_filename', default='', type=str, help='Filename of the validation data')
	parser.add_argument('--task', dest='task', default='strategy_qa', choices=['strategy_qa', 'ec_qa', 'gsm8k'], type=str, help='Dataset task to run')
	parser.add_argument('--student-model', dest='student_model', default='google/flan-t5-large', type=str,
						help='Local or hugging face path to use for the student model')
	parser.add_argument('--teacher-models', dest='teacher_models', default=['google/flan-t5-xl'], type=str, nargs='+',
						help='Local or hugging face path to use for the teacher model')
	
	parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=100, type=int, help='Maximum number of new tokens when generating answers')
	parser.add_argument('--n-beams', dest='n_beams', default=1, type=int, help='Number of beams to use in answer generation beam search')
	parser.add_argument('--n-ic-samples', dest='n_ics', default=4, type=int, help='Number of in-context samples to use for context in the student answers')
	parser.add_argument('--use-explanations', dest='use_explanations', action='store_true',
						help='Flag denoting whether student is given explanations to help understanding the problem')
	parser.add_argument('--reputation-type', dest='reputation_type', default='mm_both', type=str, help='Mental model intervention strategy')
	parser.add_argument('--max-reputation', dest='max_reputation', default=10, type=int, help='Maximum reputation level')
	parser.add_argument('--intervention-utility', dest='intervention_utility', default='mm_both', type=str, help='Mode to determine intervention utility')
	parser.add_argument('--teacher-explanation-type', dest='teacher_expl_type', default='blind_teacher_CoT', type=str, help='Teacher model explanation type')
	parser.add_argument('--student-explanation-type', dest='student_expl_type', default='cot', type=str, help='Student model explanation type')
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
	
	test_samples = task_dataset.get_test_samples() if args.task != 'strategy_qa' else task_dataset.get_validation_samples()
	train_samples = task_dataset.get_train_samples()
	print('Number of test samples = %d' % test_samples.shape[0])
	print('Number of train samples = %d' % train_samples.shape[0])
	
	budgets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
	student_model, teacher_model = None, None
	results_file = open(args.results_path, "w", encoding="utf-8-sig")
	
	for seed in [41, 42, 43]:
		
		print('Starting trials for seed: %d' % seed)
		rng_gen = default_rng(seed)
		
		print('Loading models')
		if not student_model:
			student_model, teacher_models = load_models(RNG_SEED, task_dataset.get_train_samples(), args.n_ics, args.student_model, args.teacher_models, args.task,
														args.use_explanations, args.student_expl_type, args.teacher_expl_type, args.reputation_type, args.intervention_utility,
														args.max_new_tokens, args.n_beams, args.max_reputation, args.cache_dir)
		
		else:
		
		
		print('Done')
		
		print('Computing accuracies')
		if not args.use_explanations:
			accuracy = compute_accuracy(labels, predictions_per_budget[0])
			print("Accuracy = %f\n" % accuracy)
			results_file.write("Seed = %d\n" % seed)
			results_file.write("Accuracy = %f\n" % accuracy)
			results_file.flush()
			os.fsync(results_file.fileno())
		else:
			for budget_index, budget in enumerate(budgets):
				accuracy = compute_accuracy(labels, predictions_per_budget[budget_index])
				print("Accuracy for budget %f = %f" % (budget, accuracy))
				results_file.write("Seed = %d\n" % seed)
				results_file.write("Accuracy for budget %f = %f" % (budget, accuracy))
				results_file.flush()
				os.fsync(results_file.fileno())


if __name__ == '__main__':
	main()
