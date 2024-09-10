#! /usr/bin/env python
import re

from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List


class Model:
	
	def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, model_name: str, expl_type: str, task: str, max_tokens: int, num_beams: int, samples: List[Dict]):
		
		self._model = model
		self._tokenizer = tokenizer
		self._task = task
		self._model_name = model_name
		self._expl_type = expl_type
		self._max_tokens = max_tokens
		self._num_beams = num_beams
		self._ic_samples = samples
	
	@property
	def model(self) -> PreTrainedModel:
		return self._model
	
	@property
	def tokenizer(self) -> PreTrainedTokenizer:
		return self._tokenizer

	@property
	def expl_type(self) -> str:
		return self._expl_type

	def rational_context(self, test_sample: Dict) -> str:
		context = ''
		if self._task == "strategy_qa":
			context += "\n\n".join(["Q: %s\nA: %r because %s" % (sample['question'], sample['answer'], sample['explanation']) for sample in self._ic_samples])
			context += f"\n\nQ: {test_sample['question']}\nA:"

		elif self._task == "ec_qa":
			context = "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s because %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3],
					  sample['options'][4], sample['answer'], sample['explanation'])
					 for sample in self._ic_samples])
			context += ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA:" %
			            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
			            test_sample['options'][3], test_sample['options'][4]))

		elif self._task == "gsm8k":
			context += "\n\n".join(["Q: %s\nA: %s because %s" % (sample['question'], sample['answer'], sample['explanation']) for sample in self._ic_samples])
			context += f"\n\nQ: {test_sample['question']}\nA:"

		else:
			assert False, "Dataset not recognized"
		return context

	def cot_context(self, test_sample: Dict) -> str:
		context = ''
		if self._task == 'strategy_qa':
			context += '\n\n'.join(["Q: %s\nA: %s So the answer is %r" % (ics['question'], ics['explanation'], ics['answer']) for ics in self._ic_samples])
			context += '\n\nQ: %s\nA:' % test_sample['question']
		
		elif self._task == 'ec_qa':
			context += "\n\n".join(
					['Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\n Choice 3: %s\nChoice 4: %s\n Choice 5:%s\nA: %s So the correct choice is %s' %
					 (ics['question'], ics['options'][0], ics['options'][1], ics['options'][2], ics['options'][3], ics['options'][4], ics['explanation'], ics['answer'])
					 for ics in self._ic_samples])
			context += ('\n\nQ: %s\nAnswer Choices:\n Choice 1: %s\nChoice 2: %s\n Choice 3: %s\nChoice 4: %s\n Choice 5: %s\nA:' %
			            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
			            test_sample['options'][3], test_sample['options'][4]))
		
		elif self._task == 'gsm8k':
			context += "\n\n".join(["Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			context += f"\n\nQ: {test_sample['question']}\nA:"
		
		else:
			assert False, "Task not recognized"
		
		return context
	
	def explanation_context(self, test_sample, explanation) -> str:
		context = ''
		if self._task == "strategy_qa":
			context += "\n\n".join(["Q: %s\nA: %s So the answer is %r" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			context += f"\n\nQ: %s\nA: %s So the answer is" % (test_sample['question'], explanation)
		
		elif self._task == "ec_qa":
			context += "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3],
					  sample['options'][4], sample['explanation'], sample['answer'])
					 for sample in self._ic_samples])
			context += ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is" %
			            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
			             test_sample['options'][3], test_sample['options'][4], explanation))
		
		elif self._task == 'gsm8k':
			context += "\n\n".join(["Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			context += "\n\nQ: %s\nA: %s So the answer is" % (test_sample['question'], explanation)
		
		else:
			assert False, "Dataset not recognized"
		return context
	
	def predict(self, test_sample):
		if self._expl_type == "human":
			return None, test_sample["explanation"]
		else:
			if self.expl_type == "rationalize":
				context = self.rational_context(test_sample=test_sample)
			elif self.expl_type == "cot" or self.expl_type == "useful_teacher":
				context = self.cot_context(test_sample=test_sample)
			else:
				assert False, "ToM type not supported"
			tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
			generated = self.model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens)
			output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
		
		if "llama" in self._model_name:
			output = output[len(context):]
		output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
		
		if "The correct choice is " in output:
			output = output[len("The correct choice is "):].strip()
		
		if self.expl_type == "rationalize":
			if self._task == "ec_qa":
				if output not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(test_sample["options"]):
						if choice in output:
							output = str(i + 1)
							break
			prediction = output.split(" ")[0]
			print('%s prediction = %s' % (self._model_name, prediction))
			explanation = " ".join(output.split(" ")[2:])
			print('%s explanation = %s' % (self._model_name, explanation))
		else:
			explanation = output[:output.rfind(".") + 1]
			print('%s explanation = %s' % (self._model_name, explanation))
			prediction = output.split(" ")[-1]
			if self._task == "ec_qa":
				if prediction not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(test_sample["options"]):
						if choice in output:
							prediction = str(i + 1)
							break
			elif self._task == "strategy_qa":
				if prediction not in ["no", "yes"]:
					print("Regenerating with the explanation")
					context = self.explanation_context(test_sample, explanation)
					tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
					generated = self.model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens)
					output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
					output = output[len(context):] if context in output else output
					output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
					prediction = output.split(" ")[0]
				print('%s Prediction = %s' % (self._model_name, prediction))
			elif self._task == "gsm8k":
				prediction = re.sub(r"[^0-9.]", "", prediction)
				if prediction == "" or prediction == ".":
					for word in reversed(explanation.split(" ")):
						if bool(re.search(r"\d", word)):
							prediction = re.sub(r"[^0-9.]", "", word)
							break
		
		return prediction, explanation
