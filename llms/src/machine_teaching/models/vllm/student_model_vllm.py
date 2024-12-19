#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Union, Tuple
from machine_teaching.models.vllm.model_vllm import ModelVLLM
from machine_teaching.models.model import UnidentifiedTaskError, UnidentifiedExplanationError
from vllm import LLM, SamplingParams
from pandas import DataFrame
from tqdm import tqdm
from openai import OpenAI


class StudentModel(ModelVLLM):
	
	def teacher_explanation_context(self, test_sample: Dict, teacher_explanation: str):
		if self._task == "strategy_qa":
			context = "\n\n".join([
					"Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			context += "\n\nQ: %s\nA: %s So the answer is" % (test_sample['question'], teacher_explanation)
		
		elif self._task == "ec_qa":
			context = "\n\n".join(
					["Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is %s" %
					 (sample['question'], sample['options'][0], sample['options'][1], sample['options'][2], sample['options'][3], sample['options'][4], sample['explanation'], sample['answer'])
					 for sample in self._ic_samples])
			context += ("\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA: %s So the correct choice is" %
						(test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2], test_sample['options'][3], test_sample['options'][4], teacher_explanation))
		
		elif self._task == "gsm8k":
			context = "\n\n".join(["Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			test_sample_explanation_sents = teacher_explanation.split(".")
			test_sample_partial_explanation = test_sample_explanation_sents[0] + "."
			# print("Partial explanation = %s" % test_sample_partial_explanation)
			context += "\n\nQ: %s\nA: %s" % (test_sample['question'], test_sample_partial_explanation)
		
		else:
			raise UnidentifiedTaskError("Task %s not defined for teacher explanation context" % self._task)
		
		return context
	
	def get_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False, ic_samples: List[Dict] = None):
		if not self._use_explanations:
			return self.no_explanation_context(sample, self._ic_samples)
		else:
			if intervene:
				return self.teacher_explanation_context(sample, explanation)
			elif self._explanation_type.find('cot') != -1 or (self._explanation_type.find('chain') != -1 and self._explanation_type.find('thought') != -1):
				return self.cot_context(sample, self._ic_samples)
			elif self._explanation_type.find('expl') != -1:
				return self.explanation_context(sample, self._ic_samples, explanation)
			elif self._explanation_type.find('rational') != -1:
				return self.rational_context(sample, self._ic_samples)
			elif self._explanation_type.find('no') != -1:
				return self.no_explanation_context(sample, self._ic_samples)
			else:
				raise UnidentifiedExplanationError("Explanation type '%s' not identified." % self._explanation_type)
	
	def predict_confidence(self, sample: Dict, with_explanation: bool = False, explanation: Union[List, str] = None, debug: bool = False) -> List[float]:
		# Get generation inputs
		context = self.get_context(sample, explanation=explanation)
		
		# Generate answer
		if self.local_model:
			gen_params = SamplingParams(
					temperature=self._temperature,
					top_k=self._num_beams,
					max_tokens=self._max_tokens,
					logprobs=self._n_logprobs
			)
			outputs = self.gen_model.generate(context, gen_params)

		else:
			client = OpenAI(
					base_url="http://localhost:8000/v1",
					api_key=self.api_key,
			)
			outputs = client.chat.completions.create(
					model=self.model_name,
					messages=[{'role': 'user', 'content': context}],
					max_tokens=self._max_tokens,
					logprobs=(self._n_logprobs > 0),
					top_logprobs=self._n_logprobs,
					temperature=self._temperature,
			)
			
		# Exclude extra generation from answer
		nl_id = self.gen_model.get_tokenizer().encode('\n')[1]
		nldouble_id = self.gen_model.get_tokenizer().encode('\n\n')[1]
		logprobs = outputs[0].outputs[0].logprobs
		tokens = outputs[0].outputs[0].token_ids
		nl_pos = tokens.index(nl_id) if nl_id in tokens else self._max_tokens
		nldouble_pos = tokens.index(nldouble_id) if nldouble_id in tokens else self._max_tokens
		answer_end = nl_pos if nl_pos < nldouble_pos else nldouble_pos
		tokens = tokens[:answer_end]
		logprobs = logprobs[:answer_end]
		
		if self._task == "strategy_qa":
			# Find model answer to question
			no_id = self.gen_model.get_tokenizer().encode(' no')[1]
			yes_id = self.gen_model.get_tokenizer().encode(' yes')[1]
			no_pos = tokens.index(no_id) if no_id in tokens else self._max_tokens
			yes_pos = tokens.index(yes_id) if yes_id in tokens else self._max_tokens
			answer_pos = yes_pos if yes_pos < no_pos else no_pos
			
			# Get class scores
			if answer_pos < self._max_tokens:
				answer_logprobs = Tensor([logprob.logprob for logprob in logprobs[answer_pos].values()])
				answer_tokens_alt = list(logprobs[answer_pos].keys())
				scores = softmax(answer_logprobs, dim=-1)
				yes_score, no_score = scores[answer_tokens_alt.index(yes_id)], scores[answer_tokens_alt.index(no_id)]

			else:
				yes_score = 0.0
				no_score = 0.0

			class_scores = [yes_score, no_score]
			if debug:
				print('Yes score = %s' % yes_score)
				print('No score = %s' % no_score)
		
		elif self._task == "ec_qa":
			# Find model answer to question
			opt1_id = self.gen_model.get_tokenizer().encode('1')[1]
			opt2_id = self.gen_model.get_tokenizer().encode('2')[1]
			opt3_id = self.gen_model.get_tokenizer().encode('3')[1]
			opt4_id = self.gen_model.get_tokenizer().encode('4')[1]
			opt5_id = self.gen_model.get_tokenizer().encode('5')[1]
			opt1_pos = tokens.index(opt1_id) if opt1_id in tokens else self._max_tokens
			opt2_pos = tokens.index(opt2_id) if opt2_id in tokens else self._max_tokens
			opt3_pos = tokens.index(opt3_id) if opt3_id in tokens else self._max_tokens
			opt4_pos = tokens.index(opt4_id) if opt4_id in tokens else self._max_tokens
			opt5_pos = tokens.index(opt5_id) if opt5_id in tokens else self._max_tokens
			answer_pos = min([opt1_pos, opt2_pos, opt3_pos, opt4_pos, opt5_pos])

			# Get class scores
			if answer_pos < self._max_tokens:
				answer_logprobs = Tensor([logprob.logprob for logprob in logprobs[answer_pos].values()])
				answer_tokens_alt = list(logprobs[answer_pos].keys())
				scores = softmax(answer_logprobs, dim=-1)
				opt1_score = scores[answer_tokens_alt.index(opt1_id)]
				opt2_score = scores[answer_tokens_alt.index(opt2_id)]
				opt3_score = scores[answer_tokens_alt.index(opt3_id)]
				opt4_score = scores[answer_tokens_alt.index(opt4_id)]
				opt5_score = scores[answer_tokens_alt.index(opt5_id)]

			else:
				opt1_score = opt2_score = opt3_score = opt4_score = opt5_score = 0.0

			class_scores = [opt1_score, opt2_score, opt3_score, opt4_score, opt5_score]
			if debug:
				print('Option1 score = %s' % opt1_score)
				print('Option2 score = %s' % opt2_score)
				print('Option3 score = %s' % opt3_score)
				print('Option4 score = %s' % opt4_score)
				print('Option5 score = %s' % opt5_score)
		
		else:
			raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		return class_scores
	
	def predict(self, sample: Dict, ic_samples: List[Dict] = None, debug: bool = False, expl: Union[List, str] = None, intervene: bool = False):
		context = self.get_context(sample=sample, explanation=expl, intervene=intervene, ic_samples=ic_samples)
		gen_params = SamplingParams(
				temperature=0.0,
				top_k=self._num_beams,
				max_tokens=self._max_tokens,
				logprobs=self._n_logprobs
		)
		output_text = self.gen_model.generate(context, gen_params)[0].outputs[0].text
		output_text = output_text[len(context):] if context in output_text else output_text
		output_text = output_text[:output_text.index('\n')].strip() if '\n' in output_text else output_text.strip()
		
		if self._task == "ec_qa" and "The correct choice is " in output_text:
			output_text = output_text[len("The correct choice is "):].strip()
		
		if (not self._use_explanations or
				(self._explanation_type.find("cot") == -1 and (self._explanation_type.find("chain") == -1 and self._explanation_type.find("thought") == -1))):
			if self._task == "ec_qa":
				if output_text not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(sample["options"]):
						if choice in output_text:
							output_text = str(i + 1)
							break
			prediction = output_text.split(" ")[0]
			explanation = " ".join(output_text.split(" ")[2:])
			if debug:
				print('Student Prediction = %s' % prediction)
				print('Student Explanation = %s' % explanation)
		else:
			explanation = output_text[:output_text.rfind(".") + 1] if self._task != "gsm8k" else output_text
			prediction = output_text.split(" ")[-1]
			if debug:
				print('Student Prediction = %s' % prediction)
				print('Student Explanation = %s' % explanation)
			
			if self._task == "ec_qa":
				if prediction not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(sample["options"]):
						if choice in output_text:
							prediction = str(i + 1)
							break
			
			elif self._task == "strategy_qa":
				if prediction not in ["no", "yes"]:
					if debug:
						print("Regenerating with the explanation")
					context = self.teacher_explanation_context(sample, explanation)
					output_text = self.gen_model.generate(context, gen_params)[0].outputs[0].text
					output_text = output_text[len(context):] if context in output_text else output_text
					output_text = output_text[:output_text.index('\n')].strip() if '\n' in output_text else output_text.strip()
					prediction = output_text.split(" ")[-1]
			
			elif self._task == "gsm8k":
				prediction = re.sub(r"[^0-9.]", "", prediction)
				if prediction == "" or prediction == ".":
					for word in reversed(explanation.split(" ")):
						if bool(re.search(r"\d", word)):
							prediction = re.sub(r"[^0-9.]", "", word)
							break
			
			if debug:
				print('Student Prediction = %s' % prediction)
		
		return prediction, explanation
	
	def predict_batch(self, samples: DataFrame, intervention_indexes_per_budget: List[List[int]] = None, teacher: ModelVLLM = None, debug: bool = False) -> Tuple[List, List, List]:
		labels = []
		predictions_per_budget = [[] for _ in range(len(intervention_indexes_per_budget))]
		explanations_per_budget = [[] for _ in range(len(intervention_indexes_per_budget))]
		
		for test_index, test_sample in tqdm(samples.iterrows(), desc='Student Prediction Batch', total=samples.shape[0]):
			for i, intervention_indexes in enumerate(intervention_indexes_per_budget):
				if test_index in intervention_indexes:
					_, explanation = teacher.predict(sample=test_sample.to_dict())
					prediction_student, explanation_student = self.predict(sample=test_sample.to_dict(), expl=explanation, intervene=True, debug=debug)
				else:
					prediction_student, explanation_student = self.predict(sample=test_sample.to_dict(), expl='', intervene=False, debug=debug)
				predictions_per_budget[i].append(prediction_student)
				explanations_per_budget[i].append(explanation_student)
			labels.append(test_sample['answer'])
		
		return predictions_per_budget, explanations_per_budget, labels
