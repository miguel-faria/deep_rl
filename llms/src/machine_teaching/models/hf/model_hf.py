#! /usr/bin/env python

from machine_teaching.models.model import Model
from pandas import DataFrame
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Union, Tuple


class ModelHF(Model):
	
	def __init__(self, model_name: str, samples: Union[List[Dict], Tuple] = None, gen_model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None, expl_type: str = '',
				 task: str = '', max_tokens: int = 10, num_beams: int = 1, use_explanations: bool = True):
		
		super().__init__(model_name, samples, gen_model, expl_type, task, max_tokens, num_beams, use_explanations)
		self._tokenizer = tokenizer

	@property
	def gen_model(self) -> PreTrainedModel:
		return self._gen_model
	
	@property
	def tokenizer(self) -> PreTrainedTokenizer:
		return self._tokenizer
	
	@staticmethod
	def get_answer_idx(answers: List, answer_id: Union[str, int]) -> int:
		return len(answers) - answers[-1::-1].index(answer_id) -1
	
	def get_context(self, sample: Dict, explanation: Union[List, str] = None, ic_samples: List[Dict] = None) -> str:
		raise NotImplementedError("Method 'get_context' is not implemented in base class, subclasses should implement it.")
	
	def predict_confidence(self, sample: Dict, with_expl: bool = False) -> List[float]:
		raise NotImplementedError("Method 'predict_confidence' is not implemented in the base class, subclasses should implement it.")
	
	def predict(self, sample: Dict, ic_samples: List[Dict] = None, debug: bool = False) -> Tuple[str, str]:
		raise NotImplementedError("Method 'predict' is not implemented in the base class, subclasses should implement it.")
	
	def predict_batch(self, samples: DataFrame) -> Tuple[List, List]:
		raise NotImplementedError("Method 'predict_batch' is not implemented in the base class, subclasses should implement it.")
