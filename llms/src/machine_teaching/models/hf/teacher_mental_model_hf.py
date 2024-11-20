#! /usr/bin/env python

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from machine_teaching.models.hf.teacher_model_hf import TeacherModel
from machine_teaching.models.hf.student_model_hf import StudentModel
from transformers import PreTrainedModel, PreTrainedTokenizer
from machine_teaching.models.hf.model_hf import UnidentifiedTaskError


class UnidentifiedUtilityMetricError(Exception):
	"""Raise exception for an intervention strategy type that is not defined"""
	pass


class TeacherMentalModel(TeacherModel):
	
	def __init__(self, model_name: Union[str, List[str]], intervention_samples: Union[List[Dict], Tuple] = None, gen_model: Union[PreTrainedModel, List[PreTrainedModel]] = None,
	             tokenizer: Union[PreTrainedTokenizer, List[PreTrainedTokenizer]] = None, teacher_samples: List[Dict] = None, expl_type: str = '', task: str = '', max_tokens: int = 10,
				 num_beams: int = 1, use_explanations: bool = True, utility_type: str = '', mm_type: str = 'mm_both'):
		
		super().__init__(model_name, intervention_samples, gen_model, tokenizer, expl_type, task, max_tokens, num_beams, use_explanations)
		self._teacher_samples = teacher_samples.copy()
		self._mm_type = mm_type
		self._utility_type = utility_type
	
	@property
	def teacher_samples(self) -> List[Dict]:
		return self._teacher_samples
	
	@property
	def mm_type(self) -> str:
		return self._mm_type
	
	@property
	def utility_type(self) -> str:
		return self._utility_type
	
	def get_student_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, use_answers: bool = False, debug: bool = False) -> str:
		raise NotImplementedError("Method 'get_student_context' is not implemented in teacher mental model base class, subclasses should implement it.")

	def predict_prompt(self, prompt: str, test_sample: Dict) -> Tuple:
		tokens = self.tokenizer([prompt], return_tensors="pt").to("cuda")
		generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens, output_scores=True, return_dict_in_generate=True)
		scores = softmax(generated['scores'][0], dim=-1)
		output = self.tokenizer.batch_decode(generated['sequences'], skip_special_tokens=True)[0].strip()

		if "llama" in self._model_name:
			output = output[len(prompt):]
		output = output[:output.index('\n')].strip() if '\n' in output else output.strip()

		idx = 1 if "llama" in self._model_name else 0
		option_scores = []
		if self._task == "strategy_qa":
			yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
			yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
			option_scores = [yes_score, no_score]
			# output = output.split(" ")[-1]

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

	def simulate_utility(self, sample: Dict, use_answers: bool) -> Tuple:
		
		if use_answers:
			correct_answer = sample["answer"]
		else:
			teacher_prediction, _ = self.predict(sample)
			correct_answer = teacher_prediction

		if self._mm_type.find('both') != -1:
			no_inter_context = self.get_student_context(sample, None, False, use_answers)
			no_inter_scores, no_inter_output = self.predict_prompt(no_inter_context, sample)

			# print('AI simulated answer with no intervention (Mental Model) = %s' % no_inter_output)

			inter_context = self.get_student_context(sample, None, True, use_answers)
			inter_scores, inter_output = self.predict_prompt(inter_context, sample)
			
			# print('No intervention context: ', no_inter_context, '\n')
			# print('Intervention context: ', inter_context, '\n')
			
			# print('AI simulated answer with teacher intervention (Mental Model) = %s' % inter_output)
			# print('No intervention scores: ', no_inter_scores)
			# print('Intervention scores: ', inter_scores)

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
				context = self.get_student_context(sample, None, False, use_answers)
			else:
				context = self.get_student_context(sample, None, True, use_answers)
			option_scores, output = self.predict_prompt(context, sample)

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

	def intervention_utility(self, sample: Dict, student: StudentModel, use_answers: bool) -> Union[float, Tuple]:

		if self._utility_type.find('student') != -1 and self._utility_type.find('confidence') != -1:
			if self._utility_type.find('intervention') != -1:
				if self._utility_type.find('no') != -1:
					class_scores = student.predict_confidence(sample)
				else:
					_, explanation = self.gen_model.predict(sample)
					class_scores = student.predict_confidence(sample, with_explanation=True, explanation=explanation)
				if self._task == 'strategy_qa':
					if sample["answer"] == "yes":
						return class_scores[0]
					else:
						return class_scores[1]
				elif self._task == 'ec_qa':
					return class_scores[int(sample["answer"]) - 1]
				elif self._task == 'gsm8k':
					pass
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			elif self._utility_type.find('least') != -1:
				class_scores = student.predict_confidence(sample)
				return min(class_scores)
			elif self._utility_type.find('utility') != -1 and self._utility_type.find('correct') != -1:
				_, explanation = self.gen_model.predict(sample)
				intervention_class_scores = student.predict_confidence(sample, with_explanation=True, explanation=explanation)
				no_intervention_class_scores = student.predict_confidence(sample, with_explanation=False, explanation='')
				if self._task == 'strategy_qa':
					if sample["answer"] == "yes":
						return intervention_class_scores[0] - no_intervention_class_scores[0]
					else:
						return intervention_class_scores[1] - no_intervention_class_scores[1]
				elif self._task == 'ec_qa':
					return intervention_class_scores[int(sample["answer"]) - 1] - no_intervention_class_scores[int(sample["answer"]) - 1]
				elif self._task == 'gsm8k':
					pass
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			else:
				raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)

		elif self._utility_type.find('teacher') != -1 and self._utility_type.find('confidence') != -1:
			class_scores = self.predict_confidence(sample, with_explanation=True)
			if self._task == "strategyQA":
				return class_scores[0] if sample["answer"] == "yes" else class_scores[1]
			elif self._task == "ecqa":
				return class_scores[int(sample["answer"]) - 1]
			elif self._task == 'gsm8k':
				pass
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)

		elif (self._utility_type.find('mental') != -1 and self._utility_type.find('model') != -1) or self._utility_type.find('mm') != -1:
			_, scores = self.simulate_utility(sample, use_answers)
			return scores

		else:
			raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)
