#! /usr/bin/env python

import argparse
import pandas as pd
import torch
import os

from reputation_learning.dataset_tasks_utils import ECQA, StrategyQA, GSM8k
from reputation_learning.model import UnidentifiedTaskError
from reputation_learning.teacher_model import TeacherModel
from reputation_learning.student_model import StudentModel
from reputation_learning.teacher_mental_model import UnidentifiedUtilityMetricError
from reputation_learning.teacher_interactive_mental_model import TeacherInteractiveMentalModel
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, List, Optional, Dict
from numpy.random import default_rng, Generator

RNG_SEED = 25092024


def get_teacher_model_samples(rng_gen: Generator, train_data: pd.DataFrame, student_samples: List[pd.Series], teacher_expl_type: str, num_samples: int, student_model: StudentModel,
							  teacher_model: TeacherModel = None) -> List[Dict]:
	
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


def get_mental_model_samples(rng_gen: Generator, train_data: pd.DataFrame, task: str, mental_model_type: str, max_samples: int, student_model: StudentModel,
							 teacher_model: TeacherModel) -> Tuple[List, List]:
	
	shuffle_train = train_data.sample(frac=1, random_state=rng_gen).reset_index(drop=True)
	
	if task == 'strategy_qa':
		no_intervention_samples = [[], []]
		intervention_samples = [[], []]
		num_no_intervention_samples = [0, 0]
		num_intervention_samples = [0, 0]
		for row in shuffle_train.iterrows():
			
			# Break the cycle if reached the maximum number of samples
			if sum(num_no_intervention_samples) == max_samples or sum(num_intervention_samples) == max_samples:
				break
			
			sample = row[1].to_dict()
			
			# When you have either no intervention or both intervention modes mental model add no intervention samples
			if mental_model_type.find('no') != -1 or mental_model_type.find('both') != -1:
				student_prediction, student_explanation = student_model.predict(sample=sample, expl='', intervene=False)
				if ((student_prediction == 'yes' and num_no_intervention_samples[0] == max_samples // 2) or
						(student_prediction == 'no' and num_no_intervention_samples[1] == max_samples // 2)):      # keep number of yes and no ansewers balanced
					continue
				
				no_intervention_sample = {
						"question":            	sample['question'],
						"answer":              	sample['answer'],
						"explanation":    		sample['explanation'],
						"prediction":          	student_prediction,
						"student_explanation": 	student_explanation
				}
				
				if student_prediction == 'yes':
					no_intervention_samples[0].append(no_intervention_sample)
					num_no_intervention_samples[0] += 1
				else:
					no_intervention_samples[1].append(no_intervention_sample)
					num_no_intervention_samples[1] += 1
			
			# When you have either intervention or both intervention modes mental model add intervention samples
			if mental_model_type.find('inter') != -1 or mental_model_type.find('both') != -1:
				
				_, teacher_explanation = teacher_model.predict(sample=sample)
				student_prediction, student_explanation = student_model.predict(sample=sample, expl=teacher_explanation, intervene=True)
				if ((student_prediction == 'yes' and num_intervention_samples[0] == max_samples // 2) or
						(student_prediction == 'no' and num_intervention_samples[1] == max_samples // 2)):  # keep number of yes and no ansewers balanced
					continue
				
				intervention_sample = {
						"question":            	sample['question'],
						"answer":              	sample['answer'],
						"explanation":			sample['explanation'],
						"prediction":          	student_prediction,
						"teacher_explanation": 	teacher_explanation
				}
				
				if student_prediction == 'yes':
					intervention_samples[0].append(intervention_sample)
					num_intervention_samples[0] += 1
				else:
					intervention_samples[1].append(intervention_sample)
					num_intervention_samples[1] += 1
		
		intervention_samples = intervention_samples[0] + intervention_samples[1]
		no_intervention_samples = no_intervention_samples[0] + no_intervention_samples[1]
		rng_gen.shuffle(intervention_samples)
		rng_gen.shuffle(no_intervention_samples)
		
		return no_intervention_samples, intervention_samples
	
	elif task == 'ec_qa':
		no_intervention_samples = []
		intervention_samples = []
		num_no_intervention_samples = 0
		num_intervention_samples = 0
		
		for row in shuffle_train.iterrows():
			
			# Break the cycle if reached the maximum number of samples
			if num_no_intervention_samples == max_samples or num_intervention_samples == max_samples:
				break
			sample = row[1].to_dict()
			
			# When you have either no intervention or both intervention modes mental model add no intervention samples
			if mental_model_type.find('no') != -1 or mental_model_type.find('both') != -1:
				student_prediction, student_explanation = student_model.predict(sample=sample, expl='', intervene=False)
				
				no_intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"options":              sample['options'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"student_explanation":  student_explanation
				}
				
				no_intervention_samples.append(no_intervention_sample)
				num_no_intervention_samples += 1
			
			# When you have either intervention or both intervention modes mental model add intervention samples
			if mental_model_type.find('inter') != -1 or mental_model_type.find('both') != -1:
				
				_, teacher_explanation = teacher_model.predict(sample=sample)
				student_prediction, student_explanation = student_model.predict(sample=sample, expl=teacher_explanation, intervene=True)
				
				intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"options":              sample['options'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"teacher_explanation":  teacher_explanation
				}
				
				intervention_samples.append(intervention_sample)
				num_intervention_samples += 1
		
		return no_intervention_samples, intervention_samples
	
	elif task == 'gsm8k':
		no_intervention_samples = []
		intervention_samples = []
		num_no_intervention_samples = 0
		num_intervention_samples = 0
		
		for row in shuffle_train.iterrows():
			
			# Break the cycle if reached the maximum number of samples
			if num_no_intervention_samples == max_samples or num_intervention_samples == max_samples:
				break
			sample = row[1].to_dict()
			
			# When you have either no intervention or both intervention modes mental model add no intervention samples
			if mental_model_type.find('no') != -1 or mental_model_type.find('both') != -1:
				student_prediction, student_explanation = student_model.predict(sample=sample, expl='', intervene=False)
				
				no_intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"student_explanation":  student_explanation
				}
				
				no_intervention_samples.append(no_intervention_sample)
				num_no_intervention_samples += 1
			
			# When you have either intervention or both intervention modes mental model add intervention samples
			if mental_model_type.find('inter') != -1 or mental_model_type.find('both') != -1:
				
				_, teacher_explanation = teacher_model.predict(sample=sample)
				student_prediction, student_explanation = student_model.predict(sample=sample, expl=teacher_explanation, intervene=True)
				
				intervention_sample = {
						"question":             sample['question'],
						"answer":               sample['answer'],
						"explanation":          sample['explanation'],
						"prediction":           student_prediction,
						"teacher_explanation":  teacher_explanation
				}
				
				intervention_samples.append(intervention_sample)
				num_intervention_samples += 1
		
		return no_intervention_samples, intervention_samples
	
	else:
		raise UnidentifiedTaskError('Task %s not defined.' % task)


def load_models(rng_seed: int, train_data: pd.DataFrame, num_samples: int, student_model_path: str, teacher_model_path: str, task: str, use_explanations: bool, student_expl_type: str,
				teacher_expl_type: str, mental_model_type: str, intervention_utility: str, max_tokens: int, max_student_context: int, num_beams: int,
				cache_dir: Path) -> Tuple[StudentModel, Optional[TeacherModel], Optional[TeacherInteractiveMentalModel]]:
	
	rng_gen = default_rng(rng_seed)
	
	print('Setting up the Student Model')
	train_idxs = rng_gen.choice(train_data.shape[0], num_samples, replace=False)
	student_samples = [train_data.iloc[idx].to_dict() for idx in train_idxs]
	
	student_tokenizer = AutoTokenizer.from_pretrained(student_model_path, cache_dir=cache_dir, use_fast=False)
	teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, cache_dir=cache_dir, use_fast=False) if teacher_model_path != 'human' else None
	
	if "llama" in student_model_path:
		student_gen_model = AutoModelForCausalLM.from_pretrained(student_model_path, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16)
	else:
		student_gen_model = AutoModelForSeq2SeqLM.from_pretrained(student_model_path, device_map="auto", cache_dir=cache_dir)
	
	student_model = StudentModel(student_model_path, student_samples, student_gen_model, student_tokenizer, student_expl_type, task, max_tokens, num_beams, use_explanations)
	
	if use_explanations:
		print('Setting up the Teacher Model')
		if student_expl_type.find('human') != -1:
			teacher_model = TeacherModel(teacher_model_path)
			mental_model = None
		
		else:
			if "llama" in teacher_model_path:
				teacher_gen_model = AutoModelForCausalLM.from_pretrained(teacher_model_path, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16)
			else:
				teacher_gen_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_path, device_map="auto", cache_dir=cache_dir)
			
			print('Getting teacher samples')
			teacher_samples = get_teacher_model_samples(rng_gen, train_data, student_samples, teacher_expl_type, num_samples, student_model)
			print('Creating Teacher Model')
			teacher_model = TeacherModel(teacher_model_path, teacher_samples, teacher_gen_model, teacher_tokenizer, teacher_expl_type, task, max_tokens, num_beams, use_explanations)
			
			if intervention_utility.find('mm') != -1 or (intervention_utility.find('mental') != -1 and intervention_utility.find('model') != -1):
				print('Setting up the Teacher Mental Model')
				print('Getting mental models samples')
				mm_samples = get_mental_model_samples(rng_gen, train_data, task, mental_model_type, num_samples, student_model, teacher_model)
				print('Creating Teacher Mental Model')
				mental_model = TeacherInteractiveMentalModel(teacher_model_path, mm_samples, teacher_gen_model, teacher_tokenizer, teacher_expl_type, task, max_tokens, num_beams, use_explanations,
															 intervention_utility, mental_model_type, None, max_student_context)
			
			else:
				mental_model = None
		
		return student_model, teacher_model, mental_model
	
	else:
		return student_model, None, None


def get_student_performance_per_budget(budgets: List[float], test_samples: pd.DataFrame, student: StudentModel, teacher: TeacherInteractiveMentalModel,
									   use_answers: bool, debug: bool, intervene_thresh: float = 0.5) -> Tuple[List, List]:
	
	correct_answers = []
	n_samples = test_samples.shape[0]
	n_budgets = len(budgets)
	
	assert n_budgets > 0, "There must be at least one budget value in list."
	
	max_intervention_samples = [int(budget * n_samples) for budget in budgets]
	intervention_idx_budget = [[]] * n_budgets
	n_interventions = [0] * n_budgets
	answers_budget = [[]] * n_budgets
	
	for test_idx, test_sample in test_samples.iterrows():
		
		for i in range(n_budgets):
			if n_interventions[i] >= max_intervention_samples[i]:
				prediction_student, _ = student.predict(sample=test_sample.to_dict(), expl='', intervene=False, debug=debug)
			
			else:
				_, intervene_scores = teacher.intervention_utility(test_sample.to_dict(), student, use_answers)

				if teacher.mm_type.find('both'):
					intervene_utility = intervene_scores[1] - intervene_scores[0]
				else:
					intervene_utility = intervene_scores

				if intervene_utility >= intervene_thresh:
					_, explanation = teacher.predict(sample=test_sample.to_dict())
					prediction_student, _ = student.predict(sample=test_sample.to_dict(), expl=explanation, intervene=True, debug=debug)
					intervention_idx_budget[i].append(test_idx)
					n_interventions[i] += 1
				else:
					prediction_student, _ = student.predict(sample=test_sample.to_dict(), expl='', intervene=False, debug=debug)
				
			answers_budget[i].append(prediction_student)
			
		teacher.update_student_context(test_sample.to_dict())
		correct_answers.append(test_sample['answer'])
		
	return answers_budget, correct_answers


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
	parser.add_argument('--teacher-model', dest='teacher_model', default='google/flan-t5-xl', type=str,
						help='Local or hugging face path to use for the teacher model')
	
	parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=100, type=int, help='Maximum number of new tokens when generating answers')
	parser.add_argument('--n-beams', dest='n_beams', default=1, type=int, help='Number of beams to use in answer generation beam search')
	parser.add_argument('--n-ic-samples', dest='n_ics', default=4, type=int, help='Number of in-context samples to use for context in the student answers')
	parser.add_argument('--max-student-samples', dest='max_student_samples', default=5, type=int, help='Maximum number of students to sample for mental model context')
	parser.add_argument('--use-explanations', dest='use_explanations', action='store_true',
						help='Flag denoting whether student is given explanations to help understanding the problem')
	parser.add_argument('--mm-type', dest='mm_type', default='mm_both', type=str, help='Mental model intervention strategy')
	parser.add_argument('--intervene-behaviour', dest='intervene_behaviour', default='teacher', type=str, help='Teacher intervention behaviour')
	parser.add_argument('--intervention-utility', dest='intervention_utility', default='mm_both', type=str, help='Mode to determine intervention utility')
	parser.add_argument('--teacher-explanation-type', dest='teacher_expl_type', default='blind_teacher_CoT', type=str, help='Teacher model explanation type')
	parser.add_argument('--student-explanation-type', dest='student_expl_type', default='cot', type=str, help='Student model explanation type')
	parser.add_argument('--intervention-threshold', dest='intervention_threshold', default=0.5, type=float,
						help='Threshold for intervention utility, above which the mental model gives an explanation')
	
	parser.add_argument('--use-gold-label', dest='use_gold_label', action='store_true',
						help='Flag denoting whether teacher uses the expected answers instead of its own')
	parser.add_argument('--results-path', dest='results_path', default='', type=str, help='Path to the results file')
	parser.add_argument('--debug', dest='debug', action='store_true', help='Flag that denotes the print of debug information')
	
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
	student_model, teacher_model, mental_model = None, None, None
	results_file = open(args.results_path, "w", encoding="utf-8-sig")
	
	for seed in [41, 42, 43]:
		
		print('Starting trials for seed: %d' % seed)
		rng_gen = default_rng(seed)
		
		print('Loading models')
		if not student_model:
			student_model, teacher_model, mental_model = load_models(RNG_SEED, task_dataset.get_train_samples(), args.n_ics, args.student_model, args.teacher_model, args.task,
																	 args.use_explanations, args.student_expl_type, args.teacher_expl_type, args.mm_type,
																	 args.intervention_utility, args.max_new_tokens, args.max_student_samples, args.n_beams, args.cache_dir)
		
		else:
			train_idxs = rng_gen.choice(train_samples.shape[0], args.n_ics, replace=False)
			student_samples = [train_samples.iloc[idx].to_dict() for idx in train_idxs]
			student_model.set_samples(student_samples)
			
			if args.use_explanations:
				if args.teacher_expl_type.find('human') == -1:
					teacher_samples = get_teacher_model_samples(rng_gen, train_samples, student_samples, args.teacher_expl_type, args.n_ics, student_model)
					teacher_model.set_samples(teacher_samples)
				
				if args.intervention_utility.find('mm') != -1 or (args.intervention_utility.find('mental') != -1 and args.intervention_utility.find('model') != -1):
					mm_samples = get_mental_model_samples(rng_gen, train_samples, args.task, args.mm_type, args.n_ics, student_model, teacher_model)
					mental_model.set_samples(mm_samples)
		print('Done')
		
		print('Getting student performances with interactive teacher')
		predictions_per_budget, labels = get_student_performance_per_budget(budgets, test_samples, student_model, mental_model, args.use_gold_label, args.debug, args.intervention_threshold)
		
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
