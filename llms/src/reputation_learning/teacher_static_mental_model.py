#! /usr/bin/env python

from typing import Dict, List, Union
from reputation_learning.teacher_mental_model import TeacherMentalModel
from reputation_learning.model import UnidentifiedTaskError


class TeacherStaticMentalModel(TeacherMentalModel):
	
	def get_student_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, use_answers: bool = False) -> str:
		
		context = "Simulate an AI model's answer for the given question.\n\n"
		
		if ((self.explanation_type.find('useful') != -1 and self.explanation_type.find('teacher') != -1) or
				(self.explanation_type.find('mental') != -1 and self.explanation_type.find('model') != -1)):
			if intervene:
				intervention_samples = self._ic_samples[1] if isinstance(self._ic_samples, tuple) else self._ic_samples
				_, teacher_explanation = self.predict(sample)
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
