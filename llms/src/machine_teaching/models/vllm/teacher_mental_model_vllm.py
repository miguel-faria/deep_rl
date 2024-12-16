#! /usr/bin/env python

from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Union, Tuple
from machine_teaching.models.vllm.teacher_model_vllm import TeacherModel
from machine_teaching.models.vllm.student_model_vllm import StudentModel
from vllm import LLM, SamplingParams
from machine_teaching.models.model import UnidentifiedTaskError


class TeacherMentalModel(TeacherModel):
	
	def __init__(self, model_name: Union[str, List[str]], intervention_samples: Union[List[Dict], Tuple] = None, gen_model: Union[LLM, List[LLM]] = None,
				 teacher_samples: List[Dict] = None, expl_type: str = '', task: str = '', max_tokens: int = 10, num_beams: int = 1, num_logprobs: int = 2,
				 use_explanations: bool = True, utility_type: str = '', mm_type: str = 'mm_both', local_model: bool = False, api_key: str = 'token-MtE2024'):
		
		super().__init__(model_name, intervention_samples, gen_model, expl_type, task, max_tokens, num_beams, num_logprobs, use_explanations, local_model, api_key)
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
	
	def predict_prompt(self, prompt: str, test_sample: Dict, debug: bool = False) -> Tuple:
		gen_params = SamplingParams(
				temperature=0.0,
				top_k=self._num_beams,
				max_tokens=self._max_tokens,
				logprobs=self._n_logprobs
		)
		# Generate answer
		generated = self.gen_model.generate(prompt, gen_params)
		
		# Exclude extra generation from answer
		nl_id = self.gen_model.get_tokenizer().encode('\n')[1]
		nldouble_id = self.gen_model.get_tokenizer().encode('\n\n')[1]
		logprobs = generated[0].outputs[0].logprobs
		tokens = generated[0].outputs[0].token_ids
		output = generated[0].outputs[0].text
		nl_pos = tokens.index(nl_id) if nl_id in tokens else self._max_tokens
		nldouble_pos = tokens.index(nldouble_id) if nldouble_id in tokens else self._max_tokens
		answer_end = nl_pos if nl_pos < nldouble_pos else nldouble_pos
		tokens = tokens[:answer_end]
		logprobs = logprobs[:answer_end]
		
		# Get the answer in text
		output = output[len(prompt):] if prompt in output else output
		output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
		
		if self._task == "strategy_qa":
			# Find model answer to question
			no_id = self.gen_model.get_tokenizer().encode(' no')[1]
			yes_id = self.gen_model.get_tokenizer().encode(' yes')[1]
			no_pos = tokens.index(no_id) if no_id in tokens else self._max_tokens
			yes_pos = tokens.index(yes_id) if yes_id in tokens else self._max_tokens
			answer_pos = yes_pos if yes_pos < no_pos else no_pos
			
			# Get class scores
			if answer_pos < self._max_tokens:
				answer_logprobs = Tensor([logprob.logprob for logprob in logprobs[answer_pos].values()])
				answer_tokens_alt = list(logprobs[answer_pos].keys())
				scores = softmax(answer_logprobs, dim=-1)
				yes_score, no_score = scores[answer_tokens_alt.index(yes_id)], scores[answer_tokens_alt.index(no_id)]

			else:
				yes_score = 0.0
				no_score = 0.0

			class_scores = [yes_score, no_score]
			if debug:
				print('Yes score = %s' % yes_score)
				print('No score = %s' % no_score)
		
		elif self._task == "ec_qa":
			# Find model answer to question
			opt1_id = self.gen_model.get_tokenizer().encode('1')[1]
			opt2_id = self.gen_model.get_tokenizer().encode('2')[1]
			opt3_id = self.gen_model.get_tokenizer().encode('3')[1]
			opt4_id = self.gen_model.get_tokenizer().encode('4')[1]
			opt5_id = self.gen_model.get_tokenizer().encode('5')[1]
			opt1_pos = tokens.index(opt1_id) if opt1_id in tokens else self._max_tokens
			opt2_pos = tokens.index(opt2_id) if opt2_id in tokens else self._max_tokens
			opt3_pos = tokens.index(opt3_id) if opt3_id in tokens else self._max_tokens
			opt4_pos = tokens.index(opt4_id) if opt4_id in tokens else self._max_tokens
			opt5_pos = tokens.index(opt5_id) if opt5_id in tokens else self._max_tokens
			answer_pos = min([opt1_pos, opt2_pos, opt3_pos, opt4_pos, opt5_pos])

			# Get class scores
			if answer_pos < self._max_tokens:
				answer_logprobs = Tensor([logprob.logprob for logprob in logprobs[answer_pos].values()])
				answer_tokens_alt = list(logprobs[answer_pos].keys())
				scores = softmax(answer_logprobs, dim=-1)
				opt1_score = scores[answer_tokens_alt.index(opt1_id)]
				opt2_score = scores[answer_tokens_alt.index(opt2_id)]
				opt3_score = scores[answer_tokens_alt.index(opt3_id)]
				opt4_score = scores[answer_tokens_alt.index(opt4_id)]
				opt5_score = scores[answer_tokens_alt.index(opt5_id)]

			else:
				opt1_score = opt2_score = opt3_score = opt4_score = opt5_score = 0.0

			class_scores = [opt1_score, opt2_score, opt3_score, opt4_score, opt5_score]
			if debug:
				print('Option1 score = %s' % opt1_score)
				print('Option2 score = %s' % opt2_score)
				print('Option3 score = %s' % opt3_score)
				print('Option4 score = %s' % opt4_score)
				print('Option5 score = %s' % opt5_score)
		
		else:
			raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		return class_scores, output
	
	def simulate_utility(self, sample: Dict, use_answers: bool) -> Tuple:
		
		if use_answers:
			correct_answer = sample["answer"]
		else:
			teacher_prediction, _ = self.predict(sample)
			correct_answer = teacher_prediction
		
		if self._mm_type.find('both') != -1:
			no_inter_context = self.get_student_context(sample, None, False, use_answers)
			no_inter_scores, no_inter_output = self.predict_prompt(no_inter_context, sample)
			
			inter_context = self.get_student_context(sample, None, True, use_answers)
			inter_scores, inter_output = self.predict_prompt(inter_context, sample)
			
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
					_, explanation = self.predict(sample, ic_samples=self.teacher_samples)
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
				_, explanation = self.predict(sample, ic_samples=self.teacher_samples)
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
			class_scores = self.predict_confidence(sample, with_explanation=True, ic_samples=self.teacher_samples)
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
