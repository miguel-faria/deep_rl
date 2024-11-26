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
	
	def predict_confidence(self, sample: Dict, with_explanation: bool = False, debug: bool = False) -> List[float]:
		context = self.get_context(sample)
		gen_params = SamplingParams(
				temperature=0.0,
				top_k=self._num_beams,
				max_tokens=self._max_tokens,
				logprobs=self._n_logprobs
		)
		outputs = self.gen_model.generate(context, gen_params)
		no_id = self.gen_model.get_tokenizer().encode(' no')[1]
		yes_id = self.gen_model.get_tokenizer().encode(' yes')[1]
		end_id = self.gen_model.get_tokenizer().encode(' yes')[1]
		
		generated_text = outputs[0].outputs[0].text
		logprobs = [list(logprob.values())[0] for logprob in outputs[0].outputs[0].logprobs]
		answer_end = generated_text.index('\n')
		logprobs_decoded = [logprob.decoded_token.strip() for logprob in logprobs]
		logprobs_values = Tensor([logprob.logprob for logprob in logprobs])
		
		idx = 1 if "llama" in self._model_name else 0
		if self._task == "strategy_qa":
			yes_id, no_id = self.gen_model.encode("yes")[idx], self.gen_model.encode("no")[idx]
			answer_id = 0
			generated_tokens = self.gen_model.encode(generated_text)
			if with_explanation and (yes_id in generated_tokens or no_id in generated_tokens):
				answer_id = generated_tokens.index(yes_id) - 1 if yes_id in generated_tokens else generated_tokens.index(no_id) - 1
			scores = softmax(outputs[0].outputs[0].logits[answer_id], dim=-1)
			
			yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
			
			class_scores = [yes_score, no_score]
			if debug:
				print('Yes score = %s' % yes_score)
				print('No score = %s' % no_score)
		
		elif self._task == "ec_qa":
			option_ids = [self.gen_model.encode(str(i))[idx] for i in range(1, 6)]
			generated_tokens = self.gen_model.encode(generated_text)
			
			answer_id = 0
			if with_explanation:
				for idx, option_id in enumerate(option_ids):
					if option_id in generated_tokens:
						answer_id = idx
						break
			
			# Get probabilities for each option (if available)
			logits = outputs[0].outputs[0].logits[answer_id]
			scores = softmax(logits, dim=-1)
			
			class_scores = [scores[opt_id].item() for opt_id in option_ids]
			if debug:
				print('Option1 score = %s' % class_scores[0])
				print('Option2 score = %s' % class_scores[1])
				print('Option3 score = %s' % class_scores[2])
				print('Option4 score = %s' % class_scores[3])
				print('Option5 score = %s' % class_scores[4])
		
		else:
			raise UnidentifiedTaskError('Task %s not defined' % self._task)
		
		return class_scores
	
	def predict(self, sample: Dict, ic_samples: List[Dict] = None, debug: bool = False) -> Tuple[str, str]:
		if self._explanation_type.find("human") != -1:
			return str(sample["answer"]), str(sample["explanation"])
		
		else:
			context = self.get_context(sample, explanation='', ic_samples=ic_samples)
			tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
			generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens)
			output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
			
			if "llama" in self._model_name:
				output = output[len(context):]
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
						tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
						generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens)
						output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
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
