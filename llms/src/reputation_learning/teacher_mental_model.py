#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from reputation_learning.teacher_model import TeacherModel
from reputation_learning.student_model import StudentModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from reputation_learning.model import UnidentifiedTaskError


class UnidentifiedUtilityMetricError(Exception):
	"""Raise exception for an intervention strategy type that is not defined"""
	pass


class TeacherMentalModel(TeacherModel):
	
	def __init__(self, model_name: str, intervention_samples: Union[List[Dict], Tuple] = None, gen_model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None, expl_type: str = '', task: str = '',
	             max_tokens: int = 10, num_beams: int = 1, use_explanations: bool = True, utility_type: str = '', mm_type: str = 'mm_both'):
		
		super().__init__(model_name, intervention_samples, gen_model, tokenizer, expl_type, task, max_tokens, num_beams, use_explanations)
		self._mm_type = mm_type
		self._utility_type = utility_type
	
	@property
	def mm_type(self) -> str:
		return self._mm_type
	
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

	def predict_prompt(self, prompt: str, test_sample: Dict) -> Tuple:
		tokens = self.tokenizer([prompt], return_tensors="pt").to("cuda")
		generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens)
		scores = softmax(generated['scores'][0], dim=-1)
		output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

		if "llama" in self._model_name:
			output = output[len(prompt):]
		output = output[:output.index('\n')].strip() if '\n' in output else output.strip()

		idx = 1 if "llama" in self._model_name else 0
		option_scores = []
		if self._task == "strategy_qa":
			yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
			yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
			# print(f'Yes score = {yes_score}')
			# print(f'No score = {no_score}')
			option_scores = [yes_score, no_score]

		elif self._task == "ec_qa":
			option1_id, option2_id, option3_id, option4_id, option5_id = (self.tokenizer.encode("1")[idx], self.tokenizer.encode("2")[idx],
																		  self.tokenizer.encode("3")[idx], self.tokenizer.encode("4")[idx],
																		  self.tokenizer.encode("5")[idx])
			option1_score, option2_score, option3_score, option4_score, option5_score = (scores[0][option1_id].item(), scores[0][option2_id].item(),
																						 scores[0][option3_id].item(), scores[0][option4_id].item(),
																						 scores[0][option5_id].item())

			if output not in ["1", "2", "3", "4", "5"]:
				option1_text_id, option2_text_id, option3_text_id, option4_text_id, option5_text_id = (
						self.tokenizer.encode(test_sample["options"][0].split(" ")[0])[idx],
						self.tokenizer.encode(test_sample["options"][1].split(" ")[0])[idx],
						self.tokenizer.encode(test_sample["options"][2].split(" ")[0])[idx],
						self.tokenizer.encode(test_sample["options"][3].split(" ")[0])[idx],
						self.tokenizer.encode(test_sample["options"][4].split(" ")[0])[idx])

				option1_score, option2_score, option3_score, option4_score, option5_score = (scores[0][option1_text_id].item(), scores[0][option2_text_id].item(),
																							 scores[0][option3_text_id].item(), scores[0][option4_text_id].item(),
																							 scores[0][option5_text_id].item())

			# print(f'Option1 score = {option1_score}')
			# print(f'Option2 score = {option2_score}')
			# print(f'Option3 score = {option3_score}')
			# print(f'Option4 score = {option4_score}')
			# print(f'Option5 score = {option5_score}')

			option_scores = [option1_score, option2_score, option3_score, option4_score, option5_score]

		elif self._task == "gsm8k":
			output_except_answer = " ".join(output.split(" ")[:-1])
			output_except_answer_tokens = self.tokenizer.encode(output_except_answer)
			answer_start_id = len(output_except_answer_tokens)

			digits = len(test_sample["answer"])
			answer_ids = self.tokenizer.encode(test_sample["answer"])
			# assert len(answer_ids) == digits + 2
			answer_score = 0.
			for i, answer_id in enumerate(answer_ids[0:]):
				if answer_start_id + i < len(generated['scores']):
					scores = softmax(generated['scores'][answer_start_id + i], dim=-1)
					answer_score += scores[0][answer_id].item()
			answer_score = answer_score / digits
			option_scores = [answer_score]

		else:
			raise UnidentifiedTaskError('Task %s not defined' % self._task)

		return option_scores, output

	def simulate_utility(self, test_sample: Dict, use_answers: bool) -> Tuple:
		if use_answers:
			correct_answer = test_sample["answer"]
		else:
			teacher_prediction, _ = self.predict(test_sample)
			correct_answer = teacher_prediction

		if self._mm_type.find('both') != -1:
			no_inter_context = self.get_context(test_sample, None, False, use_answers)
			no_inter_scores, no_inter_output = self.predict_prompt(no_inter_context, test_sample)

			# print('AI simulated answer with no intervention (Mental Model) = %s' % no_inter_output)

			inter_context = self.get_context(test_sample, None, True, use_answers)
			inter_scores, inter_output = self.predict_prompt(inter_context, test_sample)

			# print('AI simulated answer with teacher intervention (Mental Model) = %s' % inter_output)

			if self._task == "strategy_qa":
				if correct_answer == "yes":
					return [no_inter_output, inter_output], [no_inter_scores[0], inter_scores[0]]
				else:
					return [no_inter_output, inter_output], [no_inter_scores[1], inter_scores[1]]

			elif self._task == "ec_qa":
				if correct_answer == "1":
					return [no_inter_output, inter_output], [no_inter_scores[0], inter_scores[0]]
				elif correct_answer == "2":
					return [no_inter_output, inter_output], [no_inter_scores[1], inter_scores[1]]
				elif correct_answer == "3":
					return [no_inter_output, inter_output], [no_inter_scores[2], inter_scores[2]]
				elif correct_answer == "4":
					return [no_inter_output, inter_output], [no_inter_scores[3], inter_scores[3]]
				else:
					return [no_inter_output, inter_output], [no_inter_scores[4], inter_scores[4]]

			elif self._task == "gsm8k":
				return [no_inter_output, inter_output], [no_inter_scores[0], inter_scores[0]]

			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)

		else:
			if self._mm_type.find('no') != -1:
				context = self.get_context(test_sample, None, False, use_answers)
			else:
				context = self.get_context(test_sample, None, True, use_answers)
			option_scores, output = self.predict_prompt(context, test_sample)

			# print('AI simulated answer with %sintervention (Mental Model) = %s' % ('no ' if self._mm_intervention.find('no') != -1 else '', output))

			if self._task == "strategy_qa":
				if correct_answer == "yes":
					return output, option_scores[0]
				else:
					return output, option_scores[1]

			elif self._task == "ec_qa":
				if correct_answer == "1":
					return output, option_scores[0]
				elif correct_answer == "2":
					return output, option_scores[1]
				elif correct_answer == "3":
					return output, option_scores[2]
				elif correct_answer == "4":
					return output, option_scores[3]
				else:
					return output, option_scores[4]

			elif self._task == "gsm8k":
				return output, option_scores[0]

			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)

	def intervention_utility(self, test_sample: Dict, student: StudentModel, use_answers: bool) -> float:

		if self._utility_type.find('student') != -1 and self._utility_type.find('confidence') != -1:
			if self._utility_type.find('intervention') != -1:
				if self._utility_type.find('no') != -1:
					class_scores = student.predict_confidence(test_sample)
				else:
					_, explanation = self.gen_model.predict(test_sample)
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
				_, explanation = self.gen_model.predict(test_sample)
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
			outputs, scores = self.simulate_utility(test_sample, use_answers)
			return scores

		else:
			raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)
