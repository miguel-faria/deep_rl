#! /usr/bin/env python
import numpy as np
from pandas import DataFrame

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from reputation_learning.model import Model, UnidentifiedTaskError
from reputation_learning.student_model import StudentModel
from reputation_learning.teacher_model import TeacherModel


class UnidentifiedUtilityMetricError(Exception):
	"""Raise exception for an intervention strategy type that is not defined"""
	pass


class UnidentifiedReputationType(Exception):
	"""Raise exception for a reputation type not recognized."""
	pass


class ReuptationBasedStudent(StudentModel):
	
	def __init__(self, model_name: str, intervention_samples: Union[List[Dict], Tuple] = None, gen_model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None, teacher_models: List[TeacherModel] = None,
				 expl_type: str = '', task: str = '', max_tokens: int = 10, num_beams: int = 1, use_explanations: bool = True, utility_type: str = '', reputation_type: str = 'max_rep', max_reputation: int = 10):
		
		super().__init__(model_name, intervention_samples, gen_model, tokenizer, expl_type, task, max_tokens, num_beams, use_explanations)
		self._reputation_type = reputation_type.lower()
		self._use_reputation_type = True if self._reputation_type.find('no') == -1 else False
		self._utility_type = utility_type
		if teacher_models is None:
			self._teachers = None
			self._teacher_reputation = None
		else:
			self._teachers = teacher_models.copy()
			self._teacher_reputation = dict([(model.model_name, max_reputation // 2) for model in teacher_models])
		self._max_reputation = max_reputation
	
	@property
	def teachers(self) -> List[TeacherModel]:
		return self._teachers
	
	@property
	def teacher_names(self) -> List[str]:
		return [model.model_name for model in self._teachers]
	
	@property
	def teacher_reputation(self) -> Dict[str, float]:
		return self._teacher_reputation
	
	@property
	def reputation_type(self) -> str:
		return self._reputation_type
	
	@property
	def utility_type(self) -> str:
		return self._utility_type
	
	def update_reputation(self, teacher_correct: Dict[str, bool]):
		for teacher in teacher_correct.keys():
			if teacher_correct[teacher]:
				self._teacher_reputation[teacher] = min(self._teacher_reputation[teacher] + 1, self._max_reputation)
			
			else:
				self._teacher_reputation[teacher] = max(self._teacher_reputation[teacher] - 1, 0)
	
	def choose_teacher(self, rng_gen: np.random.Generator = None) -> Union[TeacherModel, List[float]]:
		if self._reputation_type.find('max') != -1:
			return self._teachers[np.argmax(list(self._teacher_reputation.values()))]
		elif self._reputation_type.find('prop') != -1:
			return self._teachers[rng_gen.choice(len(self._teacher_reputation.keys()), p=list(self._teacher_reputation.values()))]
		elif self._reputation_type.find('weight') != -1:
			reputation_norm = sum(self._teacher_reputation.values())
			return [val / reputation_norm for val in self._teacher_reputation.values()]
		else:
			raise UnidentifiedReputationType('Reputation type %s not recognized' % self._reputation_type)
	
	def intervention_utility(self, sample: Dict, rng_gen: np.random.Generator = None):
		
		scores = self.predict_confidence(sample, with_explanation=False, explanation='')
		
		if self._utility_type.find('least') != -1:
			return min(scores)
		
		elif self._utility_type.find('most') != -1:
			return max(scores)
		
		elif self._utility_type.find('utility') != -1 and self._utility_type.find('correct') != -1:
			if self._task == 'strategy_qa':
				return scores[0] if sample["answer"] == "yes" else scores[1]
			elif self._task == 'ec_qa':
				return scores[int(sample["answer"]) - 1]
			else:
				raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		elif self._utility_type.find('teacher') != -1:
			
			teacher = self.choose_teacher(rng_gen)
			answer_scores = []
			if self._reputation_type.find('weight') != -1:
				for teacher_model in self._teachers:
					model_name = teacher_model.model_name
					model_idx = self.teacher_names.index(model_name)
					teacher_scores = np.array(teacher_model.predict_confidence(sample, with_explanation=False)) * teacher[model_idx]
					if len(answer_scores) == 0:
						answer_scores.append(teacher_scores)
					else:
						for idx in range(len(answer_scores)):
							answer_scores[idx] += teacher_scores[idx]
			
			else:
				answer_scores = teacher.predict_confidence(sample, with_explanation=False)
			
			if self._utility_type.find('least') != -1:
				return min(answer_scores)
			elif self._utility_type.find('most') != -1:
				return max(answer_scores)
			elif self._utility_type.find('correct') != -1:
				if self._task == 'strategy_qa':
					return answer_scores[0] if sample["answer"] == "yes" else answer_scores[1]
				elif self._task == 'ec_qa':
					return answer_scores[int(sample["answer"]) - 1]
				else:
					raise UnidentifiedTaskError('Task %s not defined' % self._task)
			else:
				raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)
		
		else:
			raise UnidentifiedUtilityMetricError('Utility metric %s not defined' % self._utility_type)
