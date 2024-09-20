#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from teacher_model import TeacherModel
from student_model import StudentModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from model import UnidentifiedTaskError


class UnidentifiedUtilityMetricError(Exception):
	"""Raise exception for an intervention strategy type that is not defined"""
	pass


class TeacherMentalModel(TeacherModel):

	def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, model_name: str, expl_type: str, task: str, max_tokens: int, num_beams: int, intervention_samples: Union[List[Dict], Tuple],
	             use_explanations: bool, utility_type: str, mm_intervention: str):

		super().__init__(model, tokenizer, model_name, expl_type, task, max_tokens, num_beams, intervention_samples, use_explanations)
		self._mm_intervention = mm_intervention
		self._utility_type = utility_type

	@property
	def mm_intervention(self) -> str:
		return self._mm_intervention

	@property
	def utility_type(self) -> str:
		return self._utility_type

	def get_context(self, test_sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, use_answers: bool = False) -> str:
		context = "Simulate an AI model's answer for the given question.\n\n"
		if ((self.explanation_type.find('useful') != -1 and self.explanation_type.find('teacher') != -1) or
			(self.explanation_type.find('mental') != -1 and self.explanation_type.find('model') != -1)):
			if intervene:
				intervention_samples = self._ic_samples[0] if isinstance(self._ic_samples, tuple) else self._ic_samples
				_, teacher_explanation = self.predict(test_sample)
				print('Teacher explanation = %s' % teacher_explanation)
				if self._task == "strategy_qa":
					if not use_answers:
						context += "\n\n".join(
								["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is %s" %
								 (ic_sample['question'], ic_sample['answer'], ic_sample['teacher_explanation'], ic_sample['prediction'])
								 for ic_sample in intervention_samples])
						context += ("\n\nQ: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is" %
						            (test_sample['question'], test_sample['answer'], teacher_explanation))
					else:
						context += "\n\n".join(
								["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is %s" %
								 (ic_sample['question'], ic_sample['answer'], ic_sample['teacher_explanation'], ic_sample['prediction'])
								 for ic_sample in intervention_samples])
						context += ("\n\nQ: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is" %
						            (test_sample['question'], test_sample['answer'], teacher_explanation))
				elif self._task == "ec_qa":
					if not use_answers:
						context += "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s So the correct choice is %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2], ic_sample['options'][3],
								  ic_sample['options'][4], ic_sample['teacher_explanation'], ic_sample['prediction'])
								 for ic_sample in intervention_samples])
						context += ("\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s So the correct choice is" %
						            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3],
						             test_sample['options'][4], teacher_explanation))
					else:
						context += "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the correct choice is %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2], ic_sample['options'][3], ic_sample['options'][4],
								  ic_sample['answer'], ic_sample['teacher_explanation'], ic_sample['prediction'])
								 for ic_sample in intervention_samples])
						context += ("\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s So the correct choice is" %
						            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3],
						             test_sample['options'][4], teacher_explanation))
				elif self._task == "gsm8k":
					teacher_explanation_sents = teacher_explanation.split(".")
					teacher_partial_explanation = teacher_explanation_sents[0] + "."
					context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s So the answer is %s" % (inter_ic['question'], inter_ic['explanation'], inter_ic['answer'])
					                       for inter_ic in intervention_samples])
					context += f"\n\nQ: {test_sample['question']}\nAI Predicted Answer: {teacher_partial_explanation}"
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)

			else:
				no_intervention_samples = self._ic_samples[1] if isinstance(self._ic_samples, tuple) else self._ic_samples
				context = "Simulate an AI model's answer for the given question.\n\n"
				if self._task == "strategy_qa":
					if use_answers:
						context += "\n\n".join(
								["Q: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						context += "\n\nQ: %s\nAI Predicted Answer:" % test_sample['question']
					else:
						context += "\n\n".join(
								["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['answer'], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						context += "\n\nQ: %s\nCorrect Answer: %s\nAI Predicted Answer:" % (test_sample['question'], test_sample['answer'])
				elif self._task == "ec_qa":
					if use_answers:
						context += "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2],
								  ic_sample['options'][3], ic_sample['options'][4], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						context += (f"\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer:" %
						            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
						             test_sample['options'][3], test_sample['options'][4]))
					else:
						context += "\n\n".join(
								["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s" %
								 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2],
								  ic_sample['options'][3], ic_sample['options'][4], ic_sample['prediction'])
								 for ic_sample in no_intervention_samples])
						context += (f"\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer:" %
						            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
						             test_sample['options'][3], test_sample['options'][4]))
				elif self._task == "gsm8k":
					context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['answer']) for ic_sample in no_intervention_samples])
					context += "\n\nQ: %s\nAI Predicted Answer:" % test_sample['question']
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
		else:
			context += super().get_context(test_sample, explanation)

		return context

	def intervention_utility(self, test_sample: Dict, student: StudentModel) -> float:

		if self._utility_type.find('student') != -1 and self._utility_type.find('confidence') != -1:
			if self._utility_type.find('intervention') != -1:
				if self._utility_type.find('no') != -1:
					class_scores = student.predict_confidence(test_sample)
				else:
					_, explanation = self.model.predict(test_sample)
					class_scores = student.predict_confidence(test_sample, with_explanation=True, explanation=explanation)
				if self._task == 'strategy_qa':
					if test_sample["answer"] == "yes":
						return class_scores[0]
					else:
						return class_scores[1]
				elif self._task == 'ec_qa':
					return class_scores[int(test_sample["answer"]) - 1]
				elif self._task == 'gsm8k':
					pass
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			elif self._utility_type.find('least') != -1:
				class_scores = student.predict_confidence(test_sample)
				return min(class_scores)
			elif self._utility_type.find('utility') != -1 and self._utility_type.find('correct') != -1:
				_, explanation = self.model.predict(test_sample)
				intervention_class_scores = student.predict_confidence(test_sample, with_explanation=True, explanation=explanation)
				no_intervention_class_scores = student.predict_confidence(test_sample, with_explanation=False, explanation='')
				if self._task == 'strategy_qa':
					if test_sample["answer"] == "yes":
						return intervention_class_scores[0] - no_intervention_class_scores[0]
					else:
						return intervention_class_scores[1] - no_intervention_class_scores[1]
				elif self._task == 'ec_qa':
					return intervention_class_scores[int(test_sample["answer"]) - 1] - no_intervention_class_scores[int(test_sample["answer"]) - 1]
				elif self._task == 'gsm8k':
					pass
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			else:
				raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)
		
		elif self._utility_type.find('teacher') != -1 and self._utility_type.find('confidence') != -1:
				class_scores = self.predict_confidence(test_sample, with_explanation=True)
				if self._task == "strategyQA":
					return class_scores[0] if test_sample["answer"] == "yes" else class_scores[1]
				elif self._task == "ecqa":
					return class_scores[int(test_sample["answer"]) - 1]
				elif self._task == 'gsm8k':
					pass
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		elif (self._utility_type.find('mental') != -1 and self._utility_type.find('model') != -1) or self._utility_type.find('mm') != -1:
			
			if self._utility_type.find('no') != -1:
				pass
			
			elif self._utility_type.find('both') != -1:
				pass
			
			else:
				pass
		
		else:
			raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)
	
	def simulate_utility(self, test_sample: Dict, use_answers: bool):
		no_inter_predictions, inter_predictions = [], []
		no_inter_correct_scores, inter_correct_scores = [], []
		if use_answers:
			correct_answer = test_sample["answer"]
		else:
			teacher_prediction, _ = self.predict(test_sample)
			correct_answer = teacher_prediction
		
		if self._mm_intervention == "mm_no_inter":
			no_inter_prompt = self.prepare_prompt_no_inter(test_sample)
			option_scores, output = self.predict_util_no_inter(no_inter_prompt, test_sample)
			
			print(f'AI simulated answer with no intervention (Mental Model) = {output}')
			
			no_inter_predictions.append(output)
			if self._task == "strategyQA":
				if correct_answer == "yes":
					no_inter_correct_scores.append(option_scores[0])
				else:
					no_inter_correct_scores.append(option_scores[1])
			elif self._task == "ecqa":
				if correct_answer == "1":
					no_inter_correct_scores.append(option_scores[0])
				elif correct_answer == "2":
					no_inter_correct_scores.append(option_scores[1])
				elif correct_answer == "3":
					no_inter_correct_scores.append(option_scores[2])
				elif correct_answer == "4":
					no_inter_correct_scores.append(option_scores[3])
				else:
					no_inter_correct_scores.append(option_scores[4])
			elif self._task == "gsm8k":
				no_inter_correct_scores.append(option_scores[0])
		
		elif self._mm_intervention == "mm_inter":
			inter_prompt = self.prepare_prompt_inter(test_sample)
			option_scores, output = self.predict_util_inter(inter_prompt, test_sample)
			
			print(f'AI simulated with teacher intervention (Mental Model) = {output}')
			
			inter_predictions.append(output)
			if self._task == "strategyQA":
				if correct_answer == "yes":
					inter_correct_scores.append(option_scores[0])
				else:
					inter_correct_scores.append(option_scores[1])
			elif self._task == "ecqa":
				if correct_answer == "1":
					inter_correct_scores.append(option_scores[0])
				elif correct_answer == "2":
					inter_correct_scores.append(option_scores[1])
				elif correct_answer == "3":
					inter_correct_scores.append(option_scores[2])
				elif correct_answer == "4":
					inter_correct_scores.append(option_scores[3])
				else:
					inter_correct_scores.append(option_scores[4])
			elif self._task == "gsm8k":
				inter_correct_scores.append(option_scores[0])
		
		elif self._mm_intervention == "mm_both":
			no_inter_prompt = self.prepare_prompt_no_inter(test_sample)
			no_inter_option_scores, no_inter_output = self.predict_util_no_inter(no_inter_prompt, test_sample)
			
			print(f'AI simulated answer with no intervention (Mental Model) = {no_inter_output}')
			
			inter_prompt = self.prepare_prompt_inter(test_sample)
			inter_option_scores, inter_output = self.predict_util_inter(inter_prompt, test_sample)
			
			print(f'AI simulated answer with teacher intervention (Mental Model) = {inter_output}')
			
			no_inter_predictions.append(no_inter_output)
			inter_predictions.append(inter_output)
			
			if self._task == "strategyQA":
				if correct_answer == "yes":
					no_inter_correct_scores.append(no_inter_option_scores[0])
					inter_correct_scores.append(inter_option_scores[0])
				else:
					no_inter_correct_scores.append(no_inter_option_scores[1])
					inter_correct_scores.append(inter_option_scores[1])
			elif self._task == "ecqa":
				if correct_answer == "1":
					no_inter_correct_scores.append(no_inter_option_scores[0])
					inter_correct_scores.append(inter_option_scores[0])
				elif correct_answer == "2":
					no_inter_correct_scores.append(no_inter_option_scores[1])
					inter_correct_scores.append(inter_option_scores[1])
				elif correct_answer == "3":
					no_inter_correct_scores.append(no_inter_option_scores[2])
					inter_correct_scores.append(inter_option_scores[2])
				elif correct_answer == "4":
					no_inter_correct_scores.append(no_inter_option_scores[3])
					inter_correct_scores.append(inter_option_scores[3])
				else:
					no_inter_correct_scores.append(no_inter_option_scores[4])
					inter_correct_scores.append(inter_option_scores[4])
			elif self._task == "gsm8k":
				no_inter_correct_scores.append(no_inter_option_scores[0])
				inter_correct_scores.append(inter_option_scores[0])
		
		return no_inter_predictions, inter_predictions, no_inter_correct_scores, inter_correct_scores


