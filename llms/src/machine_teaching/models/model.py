#! /usr/bin/env python

from pandas import DataFrame
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Union, Tuple


class UnidentifiedUtilityMetricError(Exception):
	"""Raise exception for an intervention strategy type that is not defined"""
	pass


class UnidentifiedTaskError(Exception):
	"""Raise exception for a task not recognized."""
	pass


class UnidentifiedExplanationError(Exception):
	"""Raise exception for an explanation type that is not defined"""
	pass


class Model:
	
	_task: str
	_model_name: str
	_use_explanations: bool
	_max_tokens: int
	_num_beams: int
	_ic_samples: Union[List[Dict], Tuple]
	_explanation_type: str
	
	def __init__(self, model_name: str, samples: Union[List[Dict], Tuple] = None, gen_model = None, expl_type: str = '', task: str = '', max_tokens: int = 10,
				 num_beams: int = 1, use_explanations: bool = True):
		
		self._gen_model = gen_model
		self._task = task
		self._model_name = model_name
		self._use_explanations = use_explanations
		self._max_tokens = max_tokens
		self._num_beams = num_beams
		self._ic_samples = samples
		self._explanation_type = expl_type.lower()

	@property
	def model_name(self) -> str:
		return self._model_name

	@property
	def gen_model(self):
		return self._gen_model
	
	@property
	def explanation_type(self) -> str:
		return self._explanation_type
	
	@property
	def context_samples(self) -> Union[List[Dict], Tuple]:
		return self._ic_samples
	
	def set_samples(self, samples: Union[List[Dict], Tuple]) -> None:
		self._ic_samples = samples
	
	def no_explanation_context(self, test_sample, ic_samples: List[Dict]) -> str:
		if self._task == "strategy_qa":
			context = "\n\n".join(
					["Q: %s\nA: The answer is %s" % (sample['question'], sample['answer']) for sample in ic_samples])
			context += "\n\nQ: %s\nA: The answer is" % test_sample['question']
		
		elif self._task == "ec_qa":
			context = "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: The correct choice is %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3], sample['options'][4], sample['answer'])
					 for sample in ic_samples])
			context += ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: The correct choice is" %
						(test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3], test_sample['options'][4]))
		
		elif self._task == "gsm8k":
			context = "\n\n".join(["Q: %s\nA: The answer is %s" % (sample['question'], sample['answer']) for sample in ic_samples])
			context += "\n\nQ: %s\nA:" % test_sample['question']
		
		else:
			raise UnidentifiedTaskError("Task %s not recognized for no explanation context" % self._task)
		
		return context
	
	def rational_context(self, test_sample: Dict, ic_samples: List[Dict]) -> str:
		context = ''
		if self._task == "strategy_qa":
			context += "\n\n".join(["Q: %s\nA: %r because %s" % (sample['question'], sample['answer'], sample['explanation']) for sample in ic_samples])
			context += f"\n\nQ: {test_sample['question']}\nA:"
		
		elif self._task == "ec_qa":
			context = "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s because %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3],
					  sample['options'][4], sample['answer'], sample['explanation'])
					 for sample in ic_samples])
			context += ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA:" %
						(test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
						 test_sample['options'][3], test_sample['options'][4]))
		
		else:
			raise UnidentifiedTaskError("Task %s not recognized for rational context" % self._task)
		
		return context
	
	def cot_context(self, test_sample: Dict, ic_samples: List[Dict]) -> str:
		context = ''
		if self._task == 'strategy_qa':
			context += '\n\n'.join(["Q: %s\nA: %s So the answer is %s" % (ics['question'], ics['explanation'], ics['answer']) for ics in ic_samples])
			context += '\n\nQ: %s\nA:' % test_sample['question']
		
		elif self._task == 'ec_qa':
			context += "\n\n".join(
					['Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5:%s\nA: %s So the correct choice is %s' %
					 (ics['question'], ics['options'][0], ics['options'][1], ics['options'][2], ics['options'][3], ics['options'][4], ics['explanation'], ics['answer'])
					 for ics in ic_samples])
			context += ('\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA:' %
						(test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
						 test_sample['options'][3], test_sample['options'][4]))
		
		else:
			raise UnidentifiedTaskError("Task %s not recognized for chain of thought context" % self._task)
		
		return context
	
	def explanation_context(self, test_sample: Dict, ic_samples: List[Dict], explanation: str) -> str:
		context = ''
		if self._task == "strategy_qa":
			context += "\n\n".join(["Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in ic_samples])
			context += f"\n\nQ: %s\nA: %s So the answer is" % (test_sample['question'], explanation)
		
		elif self._task == "ec_qa":
			context += "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3],
					  sample['options'][4], sample['explanation'], sample['answer'])
					 for sample in ic_samples])
			context += ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is" %
						(test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
						 test_sample['options'][3], test_sample['options'][4], explanation))
		
		else:
			raise UnidentifiedTaskError("Task %s not recognized for simple explanation context" % self._task)
		
		return context

