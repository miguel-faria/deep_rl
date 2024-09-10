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
			context = (context + '\n\nQ: %s\nAnswer Choices:\n Choice 1: %s\nChoice 2: %s\n Choice 3: %s\nChoice 4: %s\n Choice 5: %s\nA:' %
			           (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3], test_sample['options'][4]))
		
		elif self._task == 'gsm8k':
			context += "\n\n".join([
					"Q: %s\nA: %s So the answer is %s" % (in_context_sample['question'], in_context_sample['explanation'], in_context_sample['answer'])
					for in_context_sample in self._ic_samples])
			context += f"\n\nQ: {test_sample['question']}\nA:"
		
		else:
			assert False, "Task not recognized"
		
		return context
	
	def explanation_context(self, test_sample, explanation) -> str:
		context = ''
		if self._task == "strategy_qa":
			context += "\n\n".join(
					[f"Q: %s\nA: %s So the answer is %s" % (in_context_sample['question'], in_context_sample['explanation'], in_context_sample['answer'])
					for in_context_sample in self._ic_samples])
			context += f"\n\nQ: %s\nA: %s So the answer is" % (test_sample['question'], explanation)
		
		elif self._task == "ec_qa":
			context += "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is %s" %
					 (in_context_sample['question'], in_context_sample['options'][0], in_context_sample['options'][1], in_context_sample['options'][2], in_context_sample['options'][3],
					  in_context_sample['options'][4], in_context_sample['explanation'], in_context_sample['answer'])
					 for in_context_sample in self._ic_samples])
			context += (context + "\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is" %
			            (test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
			             test_sample['options'][3], test_sample['options'][4], explanation))
		
		elif self._task == 'gsm8k':
			context += "\n\n".join([
					"Q: %s\nA: %s So the answer is %s" % (in_context_sample['question'], in_context_sample['explanation'], in_context_sample['answer'])
					for in_context_sample in self._ic_samples])
			context += "\n\nQ: %s\nA: %s So the answer is" % (test_sample['question'], explanation)
		
		else:
			assert False, "Dataset not recognized"
		return context
	
	def predict_single(self, test_sample):
		if self.model_name == "human":
			return None, test_sample["gold_explanation"]
		else:
			if self.expl_type == "blind_teacher_rationalize":
				context = self.prepare_context_rational(test_sample=test_sample)
			elif self.expl_type == "blind_teacher_CoT" or self.expl_type == "useful_teacher":
				context = self.prepare_context_CoT(test_sample=test_sample)
			else:
				assert False, "ToM type not supported"
			tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
			generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
			output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
		
		if "llama" in self.model_name:
			output = output[len(context):]
		output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
		
		if "The correct choice is " in output:
			output = output[len("The correct choice is "):].strip()
		
		if self.expl_type == "blind_teacher_rationalize":
			if self.dataset == "ecqa":
				if output not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(test_sample["options"]):
						if choice in output:
							output = str(i + 1)
							break
			prediction = output.split(" ")[0]
			print(f'Teacher Prediction = {prediction}')
			explanation = " ".join(output.split(" ")[2:])
			print(f'Teacher Explanation = {explanation}')
		else:
			explanation = output[:output.rfind(".") + 1]
			print(f'Teacher Explanation = {explanation}')
			prediction = output.split(" ")[-1]
			if self.dataset == "ecqa":
				if prediction not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(test_sample["options"]):
						if choice in output:
							prediction = str(i + 1)
							break
			elif self.dataset == "strategyQA":
				if prediction not in ["no", "yes"]:
					print("Regenerating with the explanation")
					context = self.prepare_context_own_explanation(test_sample, explanation)
					tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
					generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
					output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
					output = output[len(context):] if context in output else output
					output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
					prediction = output.split(" ")[0]
				print(f'Teacher Prediction = {prediction}')
			elif self.dataset == "gsm8k":
				prediction = re.sub(r"[^0-9.]", "", prediction)
				if prediction == "" or prediction == ".":
					for word in reversed(explanation.split(" ")):
						if bool(re.search(r"\d", word)):
							prediction = re.sub(r"[^0-9.]", "", word)
							break
		
		return prediction, explanation