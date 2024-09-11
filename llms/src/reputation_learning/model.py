#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Union


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
	
	@staticmethod
	def get_answer_idx(answers: List, answer_id: Union[str, int]) -> int:
		return len(answers) - answers[-1::-1].index(answer_id) -1
	
	def no_explanation_context(self, test_sample) -> str:
		if self._task == "strategy_qa":
			context = "\n\n".join(
					["Q: %s\nA: The answer is %s" % (sample['question'], sample['answer']) for sample in self._ic_samples])
			context += "\n\nQ: %s\nA: The answer is" % test_sample['question']
		
		elif self._task == "ec_qa":
			context = "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: The correct choice is %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3], sample['options'][4], sample['answer'])
					 for sample in self._ic_samples])
			context += ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: The correct choice is" %
			            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3], test_sample['options'][4]))
		
		elif self._task == "gsm8k":
			context = "\n\n".join(["Q: %s\nA: The answer is %s" % (sample['question'], sample['answer']) for sample in self._ic_samples])
			context += "\n\nQ: %s\nA:" % test_sample['question']
		
		else:
			assert False, "Dataset not recognized"
		
		return context
	
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
		
		else:
			assert False, "Dataset not recognized"
		return context
	
	def cot_context(self, test_sample: Dict) -> str:
		context = ''
		if self._task == 'strategy_qa':
			context += '\n\n'.join(["Q: %s\nA: %s So the answer is %s" % (ics['question'], ics['explanation'], ics['answer']) for ics in self._ic_samples])
			context += '\n\nQ: %s\nA:' % test_sample['question']
		
		elif self._task == 'ec_qa':
			context += "\n\n".join(
					['Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\n Choice 3: %s\nChoice 4: %s\n Choice 5:%s\nA: %s So the correct choice is %s' %
					 (ics['question'], ics['options'][0], ics['options'][1], ics['options'][2], ics['options'][3], ics['options'][4], ics['explanation'], ics['answer'])
					 for ics in self._ic_samples])
			context += ('\n\nQ: %s\nAnswer Choices:\n Choice 1: %s\nChoice 2: %s\n Choice 3: %s\nChoice 4: %s\n Choice 5: %s\nA:' %
			            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
			             test_sample['options'][3], test_sample['options'][4]))
		
		else:
			assert False, "Task not recognized"
		
		return context
	
	def explanation_context(self, test_sample, explanation) -> str:
		context = ''
		if self._task == "strategy_qa":
			context += "\n\n".join(["Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
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
		
		else:
			assert False, "Dataset not recognized"
		return context
	
	def predict_confidence(self, test_sample, with_expl=False):
		context = self.no_explanation_context(test_sample=test_sample) if not with_expl else self.cot_context(test_sample=test_sample)
		tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
		generated = self.model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens, output_scores=True, return_dict_in_generate=True)
		
		idx = 1 if "llama" in self._model_name else 0
		if self._task == "strategy_qa":
			yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
			answer_id = 0
			if with_expl:
				generated_tokens = generated[0].squeeze().tolist()
				if yes_id in generated_tokens or no_id in generated_tokens:
					answer_id = generated_tokens.index(yes_id) - 1 if yes_id in generated_tokens else generated_tokens.index(no_id) - 1
			scores = softmax(generated['scores'][answer_id], dim=-1)
			
			yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
			print('Yes score = %s' % yes_score)
			print('No score = %s' % no_score)
			class_scores = [yes_score, no_score]
		
		elif self._task == "ec_qa":
			option1_id, option2_id, option3_id, option4_id, option5_id = (self.tokenizer.encode("1")[idx], self.tokenizer.encode("2")[idx], self.tokenizer.encode("3")[idx],
			                                                              self.tokenizer.encode("4")[idx], self.tokenizer.encode("5")[idx])
			option1_text_id = self.tokenizer.encode(test_sample["options"][0].split(" ")[0])[idx]
			option2_text_id = self.tokenizer.encode(test_sample["options"][1].split(" ")[0])[idx]
			option3_text_id = self.tokenizer.encode(test_sample["options"][2].split(" ")[0])[idx]
			option4_text_id = self.tokenizer.encode(test_sample["options"][3].split(" ")[0])[idx]
			option5_text_id = self.tokenizer.encode(test_sample["options"][4].split(" ")[0])[idx]
			answer_id = 0
			found_text = False
			if with_expl:
				generated_tokens = generated[0].squeeze().tolist()
				if option1_id in generated_tokens:
					answer_id = self.get_answer_idx(generated_tokens, option1_id)
				elif option2_id in generated_tokens:
					answer_id = self.get_answer_idx(generated_tokens, option2_id)
				elif option3_id in generated_tokens:
					answer_id = self.get_answer_idx(generated_tokens, option3_id)
				elif option4_id in generated_tokens:
					answer_id = self.get_answer_idx(generated_tokens, option4_id)
				elif option5_id in generated_tokens:
					answer_id = self.get_answer_idx(generated_tokens, option5_id)
				else:
					found_text = True
					if option1_text_id in generated_tokens:
						answer_id = self.get_answer_idx(generated_tokens, option1_text_id)
					if option2_text_id in generated_tokens:
						answer_id = max(answer_id, self.get_answer_idx(generated_tokens, option2_text_id))
					if option3_text_id in generated_tokens:
						answer_id = max(answer_id, self.get_answer_idx(generated_tokens, option3_text_id))
					if option4_text_id in generated_tokens:
						answer_id = max(answer_id, self.get_answer_idx(generated_tokens, option4_text_id))
					if option5_text_id in generated_tokens:
						answer_id = max(answer_id, self.get_answer_idx(generated_tokens, option5_text_id))
			
			scores = softmax(generated['scores'][answer_id], dim=-1)
			if found_text:
				option1_score, option2_score, option3_score, option4_score, option5_score = (scores[0][option1_text_id].item(), scores[0][option2_text_id].item(), scores[0][option3_text_id].item(),
				                                                                             scores[0][option4_text_id].item(), scores[0][option5_text_id].item())
			else:
				option1_score, option2_score, option3_score, option4_score, option5_score = (scores[0][option1_id].item(), scores[0][option2_id].item(), scores[0][option3_id].item(),
				                                                                             scores[0][option4_id].item(), scores[0][option5_id].item())
			print('Option1 score = %s' % option1_score)
			print('Option2 score = %s' % option2_score)
			print('Option3 score = %s' % option3_score)
			print('Option4 score = %s' % option4_score)
			print('Option5 score = %s' % option5_score)
			class_scores = [option1_score, option2_score, option3_score, option4_score, option5_score]
		
		else:
			assert False, "Dataset not recognized"
		
		return class_scores
	
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
