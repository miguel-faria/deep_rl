#! /usr/bin/env python


from transformers.models.auto import AutoModel, AutoTokenizer
from typing import Dict, List


class TeacherModel:
	
	def __init__(self, teacher_model: AutoModel, tokenizer: AutoTokenizer, task: str, max_tokens: int, num_beams: int, samples: List[Dict]):
		
		self._teacher_model = teacher_model
		self._tokenizer = tokenizer
		self._task = task
		self._max_tokens = max_tokens
		self._num_beams = num_beams
		self._ic_samples = samples
		
	@property
	def teacher_model(self):
		return self._teacher_model
	
	@property
	def tokenizer(self):
		return self._tokenizer
	
	def get_COT_context(self, test_sample: Dict) -> str:
		
		context = ''
		if self._task == 'strategy_qa':
			context = '\n\n'.join(["Q: %s\nA: %s So the answer is %r" % (ics['question'], ics['explanation'], ics['answer']) for ics in self._ic_samples])
			context += '\n\nQ: %s\nA:' % test_sample['question']
		elif self._task == 'ec_qa':
			context = "\n\n".join(
					['Q: %s\nAnswer Choices:\n Choice 1: %s\nChoice 2: %s\n Choice 3: %s\nChoice 4: %s\n Choice 5:%s\nA: %s So the correct choice is %s' %
					 (ics['question'], ics['options'][0], ics['options'][1], ics['options'][2], ics['options'][3], ics['options'][4], ics['explanation'], ics['answer'])
					 for ics in self._ic_samples])
			context = (context + '\n\nQ: %s\nAnswer Choices:\n Choice 1: %s\nChoice 2: %s\n Choice 3: %s\nChoice 4: %s\n Choice 5: %s\nA:' %
			           (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3], test_sample['options'][4]))
			
		elif self._task == 'gsm8k':
			context = "\n\n".join([
					f"Q: {in_context_sample['question']}\nA: {in_context_sample['gold_explanation']} So the answer is {in_context_sample['answer']}"
					for in_context_sample in self.in_context_samples])
			context += f"\n\nQ: {test_sample['question']}\nA:"
			
		else:
			assert False, "Task not recognized"
			
		return context
