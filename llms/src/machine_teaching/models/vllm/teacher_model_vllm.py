#! /usr/bin/env python
import re

from pandas import DataFrame
from torch.nn.functional import softmax
from torch import Tensor
from typing import Dict, List, Tuple, Union
from machine_teaching.models.vllm.model_vllm import ModelVLLM
from machine_teaching.models.model import UnidentifiedTaskError, UnidentifiedExplanationError
from vllm import LLM, SamplingParams
from tqdm import tqdm


class TeacherModel(ModelVLLM):
	
	def __init__(self, model_name: str, samples: List[Dict] = None, gen_model: LLM = None, expl_type: str = '', task: str = '', max_tokens: int = 10, num_beams: int = 1,
				 num_logprobs: int = 2, use_explanations: bool = True):
		
		super().__init__(model_name, samples, gen_model, expl_type, task, max_tokens, num_beams, num_logprobs, use_explanations)
	
	def get_context(self, sample: Dict, explanation: Union[List, str] = None, ic_samples: List[Dict] = None) -> str:
		if ic_samples is None:
			ic_samples = self._ic_samples[0] if isinstance(self._ic_samples, tuple) else self._ic_samples
		if not self._use_explanations:
			return self.no_explanation_context(sample, ic_samples)
		else:
			if self._explanation_type.find('blind') != -1:
				if self._explanation_type.find('rational') != -1:
					return self.rational_context(sample, ic_samples)
				elif self._explanation_type.find('cot') != -1 or (self._explanation_type.find('chain') != -1 and self._explanation_type.find('thought') != -1):
					return self.cot_context(sample, ic_samples)
			elif self._explanation_type.find('useful') != -1:
				return self.cot_context(sample, ic_samples)
			elif self._explanation_type.find('expl') != -1:
				return self.explanation_context(sample, ic_samples, explanation)
			else:
				raise UnidentifiedExplanationError("Explanation type '%s' not identified." % self._explanation_type)
	
	def predict_confidence(self, sample: Dict, with_explanation: bool = False, debug: bool = False, ic_samples: List[Dict] = None) -> List[float]:
		context = self.get_context(sample, explanation='', ic_samples=ic_samples)
		gen_params = SamplingParams(
				temperature=0.0,
				top_k=self._num_beams,
				max_tokens=self._max_tokens,
				logprobs=self._n_logprobs
		)

		# Generate answer
		outputs = self.gen_model.generate(context, gen_params)

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
		
		idx = 1 if "llama" in self._model_name else 0
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
	
	def predict(self, sample: Dict, ic_samples: List[Dict] = None, debug: bool = False) -> Tuple[str, str]:
		if self._explanation_type.find("human") != -1:
			return str(sample["answer"]), str(sample["explanation"])
		
		else:
			context = self.get_context(sample, explanation='', ic_samples=ic_samples)
			gen_params = SamplingParams(
					temperature=0.0,
					top_k=self._num_beams,
					max_tokens=self._max_tokens,
					logprobs=self._n_logprobs
			)
			output = self.gen_model.generate(context, gen_params)[0].outputs[0].text
			output = output[len(context):] if context in output else output
			output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
			
			if "The correct choice is " in output:
				output = output[len("The correct choice is "):].strip()
			
			if self._explanation_type.find('rationalize') != -1:
				if self._task == "ec_qa":
					if output not in ["1", "2", "3", "4", "5"]:
						for i, choice in enumerate(sample["options"]):
							if choice in output:
								output = str(i + 1)
								break
				prediction = output.split(" ")[0]
				explanation = " ".join(output.split(" ")[2:])
				if debug:
					print('%s prediction = %s' % (self._model_name, prediction))
					print('%s explanation = %s' % (self._model_name, explanation))
			
			else:
				explanation = output[:output.rfind(".") + 1]
				if debug:
					print('%s explanation = %s' % (self._model_name, explanation))
				prediction = output.split(" ")[-1]
				if self._task == "ec_qa":
					if prediction not in ["1", "2", "3", "4", "5"]:
						for i, choice in enumerate(sample["options"]):
							if choice in output:
								prediction = str(i + 1)
								break
				
				elif self._task == "strategy_qa":
					if prediction not in ["no", "yes"]:
						if debug:
							print("Regenerating with the explanation")
						context_samples = self._ic_samples[0] if isinstance(self._ic_samples, tuple) else self._ic_samples
						context = self.explanation_context(sample, context_samples, explanation)
						output = self.gen_model.generate(context, gen_params)[0].outputs[0].text
						output = output[len(context):] if context in output else output
						output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
						prediction = output.split(" ")[0]
					
					if debug:
						print('%s Prediction = %s' % (self._model_name, prediction))
				
				elif self._task == "gsm8k":
					prediction = re.sub(r"[^0-9.]", "", prediction)
					if prediction == "" or prediction == ".":
						for word in reversed(explanation.split(" ")):
							if bool(re.search(r"\d", word)):
								prediction = re.sub(r"[^0-9.]", "", word)
								break
			
			return prediction, explanation
	
	def predict_batch(self, samples: DataFrame, debug: bool = False) -> Tuple[List, List]:
		predictions = []
		explanations = []
		
		for test_index, test_sample in tqdm(samples.iterrows(), desc='Teacher Prediction Batch', total=samples.shape[0]):
			prediction, explanation = self.predict(sample=test_sample.to_dict(), debug=debug)
			predictions.append(prediction)
			explanations.append(explanation)
		
		return predictions, explanations
