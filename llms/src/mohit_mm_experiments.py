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
from reputation_learning.teacher_static_mental_model import TeacherStaticMentalModel
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, List, Optional, Dict
from numpy.random import default_rng, Generator
from tqdm import tqdm


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
			
			student_prediction_no_intervene, _ = student_model.predict(sample, expl='', intervene=False)  # get student prediction without teacher intervention
			
			teacher_expl = sample['explanation'] if teacher_model is None else teacher_model.predict(sample)[0]
			student_prediction_intervene, _ = student_model.predict(sample, expl=teacher_expl, intervene=True)  # get student prediction with teacher intervention
			
			if student_prediction_intervene == sample['answer'] and student_prediction_no_intervene != student_prediction_intervene:  # add sample if the intervention made student right
				teacher_samples.append(sample)
			
			idx += 1
	
	else:
		samples_idxs = rng_gen.choice(train_data.shape[0], num_samples, replace=False)
		teacher_samples = [train_data.iloc[x].to_dict() for x in samples_idxs]
	
	return teacher_samples


def get_mental_model_samples(rng_gen: Generator, train_data: pd.DataFrame, task: str, mental_model_type: str, max_samples: int,
							 student_model: StudentModel, teacher_model: TeacherModel) -> Tuple[List, List]:
	
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
				teacher_expl_type: str, mental_model_type: str, intervention_utility: str, max_tokens: int, num_beams: int, cache_dir: Path) -> Tuple[StudentModel, Optional[TeacherModel], Optional[TeacherStaticMentalModel]]:
	
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
				mental_model = TeacherStaticMentalModel(teacher_model_path, mm_samples, teacher_gen_model, teacher_tokenizer, teacher_samples, teacher_expl_type, task, max_tokens,
														num_beams, use_explanations, intervention_utility, mental_model_type)
			
			else:
				mental_model = None
		
		return student_model, teacher_model, mental_model
	
	else:
		return student_model, None, None


def get_intervention_idx_budget(student_model: StudentModel, mental_model: TeacherStaticMentalModel, rng_gen: Generator, budgets: List[float], test_samples: pd.DataFrame,
								intervention_utility: str, use_explanation: bool, use_answers: bool, deceive: bool) -> Tuple:
	intervention_idx_budget = []
	intercention_conf_budget = []
	n_test_samples = test_samples.shape[0]
	
	if use_explanation:
		if intervention_utility.find('random') != -1:
			for budget in budgets:
				budget_count = int(budget * n_test_samples)
				sample_indexes = list(rng_gen.choice(test_samples.shape[0], budget_count, replace=False).astype(int))
				intervention_idx_budget.append(sample_indexes)
		
		elif intervention_utility.find('oracle') != -1:
			sample_indexes = []
			for row in test_samples.iterrows():
				idx, sample = row
				prediction, _ = student_model.predict(sample=sample.to_dict(), expl='', intervene=False)
				if prediction != sample['answer']:
					sample_indexes.append(idx)
			for _ in budgets:
				intervention_idx_budget.append(sample_indexes)
		
		else:
			sample_confidence_pairs = []
			for idx, sample in tqdm(test_samples.iterrows(), desc='Test Samples', total=test_samples.shape[0]):
				confidence_scores = mental_model.intervention_utility(sample.to_dict(), student_model, use_answers)
				sample_confidence_pairs.append((idx, confidence_scores))
			
			if intervention_utility.find('student') != -1 and intervention_utility.find('confidence') != -1:
				if intervention_utility == "intervention_correct_student_confidence" or intervention_utility == "utility_correct_student_confidence":
					sample_confidence_pairs = sorted(sample_confidence_pairs, key=lambda x: x[1], reverse=True)
				else:
					sample_confidence_pairs = sorted(sample_confidence_pairs, key=lambda x: x[1])
				
				for budget in budgets:
					budget_count = int(budget * len(test_samples))
					intervention_samples = [sample_confidence_pair[0] for sample_confidence_pair in sample_confidence_pairs][:budget_count]
					intervention_confs = [sample_confidence_pair[1] for sample_confidence_pair in sample_confidence_pairs][:budget_count]
					intervention_idx_budget.append(intervention_samples)
					intercention_conf_budget.append(intervention_confs)
			
			elif intervention_utility.find('teacher') != -1 and intervention_utility.find('confidence') != -1:
				sample_confidence_pairs = sorted(sample_confidence_pairs, key=lambda x: x[1], reverse=True)
				for budget in budgets:
					budget_count = int(budget * len(test_samples))
					intervention_samples = [sample_confidence_pair[0] for sample_confidence_pair in sample_confidence_pairs][:budget_count]
					intervention_confs = [sample_confidence_pair[1] for sample_confidence_pair in sample_confidence_pairs][:budget_count]
					intervention_idx_budget.append(intervention_samples)
					intercention_conf_budget.append(intervention_confs)
			
			elif (intervention_utility.find('mental') != -1 and intervention_utility.find('model') != -1) or intervention_utility.find('mm') != -1:
				if intervention_utility.find('both') != -1:
					sample_utility_correct_scores = sorted(sample_confidence_pairs, key=lambda x: (x[1][1] - x[1][0]), reverse=True) if not deceive else sorted(sample_confidence_pairs, key=lambda x: (x[1][1] - x[1][0]))
					for budget in budgets:
						budget_count = int(budget * len(test_samples))
						intervention_samples = [sample_utility_correct_score[0] for sample_utility_correct_score in sample_utility_correct_scores][:budget_count]
						intervention_confs = [sample_utility_correct_score[1] for sample_utility_correct_score in sample_utility_correct_scores][:budget_count]
						intervention_idx_budget.append(intervention_samples)
						intercention_conf_budget.append(intervention_confs)
				
				elif intervention_utility.find('no') != -1:
					sample_no_inter_correct_scores = sorted(sample_confidence_pairs, key=lambda x: x[1])
					for budget in budgets:
						budget_count = int(budget * len(test_samples))
						intervention_samples = [sample_no_inter_correct_score[0] for sample_no_inter_correct_score in sample_no_inter_correct_scores][:budget_count]
						intervention_confs = [sample_no_inter_correct_score[1] for sample_no_inter_correct_score in sample_no_inter_correct_scores][:budget_count]
						intervention_idx_budget.append(intervention_samples)
						intercention_conf_budget.append(intervention_confs)
				
				else:
					sample_inter_correct_scores = sorted(sample_confidence_pairs, key=lambda x: x[1], reverse=True)
					for budget in budgets:
						budget_count = int(budget * len(test_samples))
						intervention_samples = [sample_inter_correct_score[0] for sample_inter_correct_score in sample_inter_correct_scores][:budget_count]
						intervention_confs = [sample_inter_correct_score[1] for sample_inter_correct_score in sample_inter_correct_scores][:budget_count]
						intervention_idx_budget.append(intervention_samples)
						intercention_conf_budget.append(intervention_confs)
			
			else:
				raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % intervention_utility)
	
	return intervention_idx_budget, intercention_conf_budget


def compute_accuracy(labels, predictions):
	correct = 0
	for (label, prediction) in zip(labels, predictions):
		if label == prediction:
			correct += 1
	
	return correct / len(labels)


def main( ):
	parser = argparse.ArgumentParser(description='Machine teaching with Theory of Mind based mental models experiments from Mohit Bensal')
	# Models arguments
	parser.add_argument('--cache-dir', dest='cache_dir', default='', type=str, help='Path to the cache directory, where downladed models are stored')
	parser.add_argument('--train-filename', dest='train_filename', default='', type=str, help='Filename of the training data')
	parser.add_argument('--test-filename', dest='test_filename', default='', type=str, help='Filename of the testing data')
	parser.add_argument('--val-filename', dest='val_filename', default='', type=str, help='Filename of the validation data')
	parser.add_argument('--task', dest='task', default='strategy_qa', choices=['strategy_qa', 'ec_qa', 'gsm8k'], type=str, help='Dataset task to run')
	parser.add_argument('--student-model', dest='student_model', default='google/flan-t5-large', type=str,
						help='Local or hugging face path to use for the student model')
	parser.add_argument('--teacher-model', dest='teacher_model', default='google/flan-t5-xl', type=str,
						help='Local or hugging face path to use for the teacher model')
	parser.add_argument('--use-gold-label', dest='use_gold_label', action='store_true',
						help='Flag denoting whether teacher uses the expected answers instead of its own')

	# Execution arguments
	parser.add_argument('--budgets', dest='budgets', default=None, type=float, nargs='+',
	                    help='Interaction budgets to test the teaching. Default: [0, 0.2, 0.4, 0.6, 0.8, 1.0]')
	parser.add_argument('--data-dir', dest='data_dir', default='', type=str, help='Path to the directory with the datasets')
	parser.add_argument('--max-new-tokens', dest='max_new_tokens', default=100, type=int, help='Maximum number of new tokens when generating answers')
	parser.add_argument('--n-beams', dest='n_beams', default=1, type=int, help='Number of beams to use in answer generation beam search')
	parser.add_argument('--n-ic-samples', dest='n_ics', default=4, type=int, help='Number of in-context samples to use for context in the student answers')
	parser.add_argument('--use-explanations', dest='use_explanations', action='store_true',
						help='Flag denoting whether student is given explanations to help understanding the problem')
	parser.add_argument('--mm-type', dest='mm_type', default='mm_both', type=str, help='Mental model intervention strategy')
	parser.add_argument('--intervene-behaviour', dest='intervene_behaviour', default='teacher', type=str, help='Teacher intervention behaviour')
	parser.add_argument('--intervention-utility', dest='intervention_utility', default='mm_both', type=str, help='Mode to determine intervention utility')
	parser.add_argument('--teacher-explanation-type', dest='teacher_expl_type', default='blind_teacher_CoT', type=str, help='Teacher model explanation type')
	parser.add_argument('--student-explanation-type', dest='student_expl_type', default='cot', type=str, help='Student model explanation type')
	parser.add_argument('--deceive', dest='deceive', action='store_true', help='Flag denoting whether teacher gives deceiving explanations')
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
	
	budgets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if args.budgets is None else args.budgets
	student_model, teacher_model, mental_model = None, None, None
	results_file = open(args.results_path, "w", encoding="utf-8-sig")
	
	for seed in [41, 42, 43]:
		
		print('Starting trials for seed: %d' % seed)
		rng_gen = default_rng(seed)
		
		print('Loading models')
		if not student_model:
			student_model, teacher_model, mental_model = load_models(RNG_SEED, task_dataset.get_train_samples(), args.n_ics, args.student_model, args.teacher_model, args.task,
																	 args.use_explanations, args.student_expl_type, args.teacher_expl_type, args.mm_type,
																	 args.intervention_utility, args.max_new_tokens, args.n_beams, args.cache_dir)
		
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
		
		print('Getting samples for intervention')
		intervention_idxs_per_budget, intervention_conf_budget = get_intervention_idx_budget(student_model, mental_model, rng_gen, budgets, test_samples, args.intervention_utility,
																							 args.use_explanations, args.use_gold_label, args.deceive)
		
		print(intervention_conf_budget)
		print('Intervention utilities:')
		for i in range(len(budgets)):
			print('Budget %f: max = %f\tmin = %f' % (budgets[i], max(intervention_conf_budget[i]), min(intervention_conf_budget[i])))
		
		print('Getting predictions for each budget level')
		predictions_per_budget, _, labels = student_model.predict_batch(test_samples, intervention_idxs_per_budget, teacher_model)
		
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
