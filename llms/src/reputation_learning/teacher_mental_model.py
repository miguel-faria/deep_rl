#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from teacher_model import TeacherModel
from student_model import StudentModel
from transformers import PreTrainedModel, PreTrainedTokenizer


class TeacherMentalModel(TeacherModel):

	def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, model_name: str, expl_type: str, task: str, max_tokens: int, num_beams: int, samples: List[Dict],
	             use_explanations: bool, utility_type: str, mm_intervention: str):

		super().__init__(model, tokenizer, model_name, expl_type, task, max_tokens, num_beams, samples, use_explanations)
		self._mm_intervention = mm_intervention
		self._utility_type = utility_type

	@property
	def mm_intervention(self) -> str:
		return self._mm_intervention

	@property
	def utility_type(self) -> str:
		return self._utility_type

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
			elif self._utility_type.find('least') != -1:
				class_scores = student.predict_confidence(test_sample)
				return min(class_scores)
			elif self._utility_type.find('utility') != -1 and self._utility_type.find('correct') != -1:
				pass




