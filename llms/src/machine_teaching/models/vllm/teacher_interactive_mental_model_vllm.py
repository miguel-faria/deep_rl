#! /usr/bin/env python

from typing import Dict, List, Union, Tuple
from machine_teaching.models.vllm.teacher_mental_model_vllm import TeacherMentalModel
from machine_teaching.models.model import UnidentifiedTaskError
from vllm import LLM


class TeacherInteractiveMentalModel(TeacherMentalModel):
	
	def __init__(self, model_name: Union[str, List[str]], intervention_samples: Union[List[Dict], Tuple] = None, gen_model: Union[LLM, List[LLM]] = None,
				 teacher_samples: List[Dict] = None, expl_type: str = '', task: str = '', max_tokens: int = 10, num_beams: int = 1, num_logprobs: int = 2,
				 use_explanations: bool = True, utility_type: str = '', mm_type: str = 'mm_both', student_context: List[Dict] = None, max_student_context: int = 5,
				 local_model: bool = True, temperature: float = 0.0, api_key: str = 'token-MtE2024', model_url: str = "http://localhost:8000/v1"):
		
		super().__init__(model_name, intervention_samples, gen_model, teacher_samples, expl_type, task, max_tokens, num_beams, num_logprobs, use_explanations, utility_type,
		                 mm_type, local_model, temperature, api_key, model_url)
		if student_context is None:
			self._student_context = []
		else:
			self._student_context = student_context.copy()
		self._max_student_context = max_student_context
	
	@property
	def student_context(self) -> List[Dict]:
		return self._student_context
	
	def update_student_context(self, new_context_sample: Dict) -> None:
		if len(self._student_context) < self._max_student_context:
			self._student_context.append(new_context_sample)
		else:
			self._student_context.pop(0)
			self._student_context.append(new_context_sample)
	
	def reset_student_context(self) -> None:
		self._student_context = []
	
	def get_student_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, use_answers: bool = False, debug: bool = False) -> str:
		
		context = "\n\nSimulate an AI model's answer for the given question.\n\n"
		
		if len(self._student_context) > 0:
			if self._task == "strategy_qa":
				if not use_answers:
					context += "\n\n".join(
							["Q: %s\nAI Predicted Answer: %s So the answer is %s" %
							 (sample['question'], sample['explanation'], sample['prediction'])
							 for sample in self._student_context])
				else:
					context += "\n\n".join(
							["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is %s" %
							 (sample['question'], sample['answer'], sample['explanation'], sample['prediction'])
							 for sample in self._student_context])
			
			elif self._task == "ec_qa":
				if not use_answers:
					context += "\n\n".join(
							["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s So the correct choice is %s" %
							 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3],
							  sample['options'][4], sample['explanation'], sample['prediction'])
							 for sample in self._student_context])
				else:
					context += "\n\n".join(
							["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the correct choice is %s" %
							 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3], sample['options'][4],
							  sample['answer'], sample['explanation'], sample['prediction'])
							 for sample in self._student_context])
			
			elif self._task == "gsm8k":
				context += "\n\n".join(["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is %s" % (sample['question'], sample['answer'], sample['explanation'], sample['prediction'])
										for sample in self._student_context])
			
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		else:
			if ((self.explanation_type.find('useful') != -1 and self.explanation_type.find('teacher') != -1) or
					(self.explanation_type.find('mental') != -1 and self.explanation_type.find('model') != -1)):
				
				if intervene:
					intervention_samples = self._ic_samples[1] if isinstance(self._ic_samples, tuple) else self._ic_samples
					_, teacher_explanation = self.predict(sample, ic_samples=self.teacher_samples)
					if debug:
						print('Teacher explanation = %s' % teacher_explanation)
					if self._task == "strategy_qa":
						if not use_answers:
							context += "\n\n".join(
									["Q: %s\nAI Predicted Answer: %s So the answer is %s" %
									 (ic_sample['question'], ic_sample['teacher_explanation'], ic_sample['prediction'])
									 for ic_sample in intervention_samples])
							context += ("\n\nQ: %s\nAI Predicted Answer: %s So the answer is" %
										(sample['question'], teacher_explanation))
						else:
							context += "\n\n".join(
									["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is %s" %
									 (ic_sample['question'], ic_sample['answer'], ic_sample['teacher_explanation'], ic_sample['prediction'])
									 for ic_sample in intervention_samples])
							context += ("\n\nQ: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the answer is" %
										(sample['question'], sample['answer'], teacher_explanation))
					
					elif self._task == "ec_qa":
						if not use_answers:
							context += "\n\n".join(
									["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s So the correct choice is %s" %
									 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2], ic_sample['options'][3],
									  ic_sample['options'][4], ic_sample['teacher_explanation'], ic_sample['prediction'])
									 for ic_sample in intervention_samples])
							context += ("\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s So the correct choice is" %
										(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3],
										 sample['options'][4], teacher_explanation))
						else:
							context += "\n\n".join(
									["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nCorrect Answer: %s\nAI Predicted Answer: %s So the correct choice is %s" %
									 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2], ic_sample['options'][3], ic_sample['options'][4],
									  ic_sample['answer'], ic_sample['teacher_explanation'], ic_sample['prediction'])
									 for ic_sample in intervention_samples])
							context += ("\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s So the correct choice is" %
										(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3],
										 sample['options'][4], teacher_explanation))
					
					elif self._task == "gsm8k":
						teacher_explanation_sents = teacher_explanation.split(".")
						teacher_partial_explanation = teacher_explanation_sents[0] + "."
						context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s So the answer is %s" % (inter_ic['question'], inter_ic['explanation'], inter_ic['answer'])
											   for inter_ic in intervention_samples])
						context += f"\n\nQ: {sample['question']}\nAI Predicted Answer: {teacher_partial_explanation}"
					
					else:
						raise UnidentifiedTaskError('Task %s not defined' % self._task)
				
				else:
					no_intervention_samples = self._ic_samples[0] if isinstance(self._ic_samples, tuple) else self._ic_samples
					context = "Simulate an AI model's answer for the given question.\n\n"
					if self._task == "strategy_qa":
						if not use_answers:
							context += "\n\n".join(
									["Q: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['prediction'])
									 for ic_sample in no_intervention_samples])
							context += "\n\nQ: %s\nAI Predicted Answer:" % sample['question']
						else:
							context += "\n\n".join(
									["Q: %s\nCorrect Answer: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['answer'], ic_sample['prediction'])
									 for ic_sample in no_intervention_samples])
							context += "\n\nQ: %s\nCorrect Answer: %s\nAI Predicted Answer:" % (sample['question'], sample['answer'])
					
					elif self._task == "ec_qa":
						if not use_answers:
							context += "\n\n".join(
									["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s" %
									 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2],
									  ic_sample['options'][3], ic_sample['options'][4], ic_sample['prediction'])
									 for ic_sample in no_intervention_samples])
							context += (f"\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer:" %
										(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2],
										 sample['options'][3], sample['options'][4]))
						else:
							context += "\n\n".join(
									["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer: %s" %
									 (ic_sample['question'], ic_sample['options'][0], ic_sample['options'][1], ic_sample['options'][2],
									  ic_sample['options'][3], ic_sample['options'][4], ic_sample['prediction'])
									 for ic_sample in no_intervention_samples])
							context += (f"\n\nQ: %s\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nAI Predicted Answer:" %
										(sample['question'], sample['options'][0], sample['options'][1], sample['options'][2],
										 sample['options'][3], sample['options'][4]))
					
					elif self._task == "gsm8k":
						context = "\n\n".join(["Q: %s\nAI Predicted Answer: %s" % (ic_sample['question'], ic_sample['answer']) for ic_sample in no_intervention_samples])
						context += "\n\nQ: %s\nAI Predicted Answer:" % sample['question']
					
					else:
						raise UnidentifiedTaskError('Task %s not defined' % self._task)
			
			else:
				context += super().get_context(sample, explanation)
		
		return context
