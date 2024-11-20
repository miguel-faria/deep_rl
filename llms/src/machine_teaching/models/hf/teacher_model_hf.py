#! /usr/bin/env python
import re

from pandas import DataFrame
from torch.nn.functional import softmax
from typing import Dict, List, Tuple, Union
from machine_teaching.models.hf.model_hf import ModelHF, UnidentifiedTaskError, UnidentifiedExplanationError
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm


class TeacherModel(ModelHF):
	
	def __init__(self, model_name: str, samples: List[Dict] = None, gen_model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None, expl_type: str = '', task: str = '', max_tokens: int = 10, num_beams: int = 1,
				 use_explanations: bool = True):
		
		super().__init__(model_name, samples, gen_model, tokenizer, expl_type, task, max_tokens, num_beams, use_explanations)
	
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
		tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
		generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens, output_scores=True, return_dict_in_generate=True)
		
		idx = 1 if "llama" in self._model_name else 0
		if self._task == "strategy_qa":
			yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
			answer_id = 0
			if with_explanation:
				generated_tokens = generated[0].squeeze().tolist()
				if yes_id in generated_tokens or no_id in generated_tokens:
					answer_id = generated_tokens.index(yes_id) - 1 if yes_id in generated_tokens else generated_tokens.index(no_id) - 1
			scores = softmax(generated['scores'][answer_id], dim=-1)
			
			yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
			
			class_scores = [yes_score, no_score]
			if debug:
				print('Yes score = %s' % yes_score)
				print('No score = %s' % no_score)
		
		elif self._task == "ec_qa":
			option1_id, option2_id, option3_id, option4_id, option5_id = (self.tokenizer.encode("1")[idx], self.tokenizer.encode("2")[idx], self.tokenizer.encode("3")[idx],
																		  self.tokenizer.encode("4")[idx], self.tokenizer.encode("5")[idx])
			option1_text_id = self.tokenizer.encode(sample["options"][0].split(" ")[0])[idx]
			option2_text_id = self.tokenizer.encode(sample["options"][1].split(" ")[0])[idx]
			option3_text_id = self.tokenizer.encode(sample["options"][2].split(" ")[0])[idx]
			option4_text_id = self.tokenizer.encode(sample["options"][3].split(" ")[0])[idx]
			option5_text_id = self.tokenizer.encode(sample["options"][4].split(" ")[0])[idx]
			answer_id = 0
			found_text = False
			if with_explanation:
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
			
			class_scores = [option1_score, option2_score, option3_score, option4_score, option5_score]
			if debug:
				print('Option1 score = %s' % option1_score)
				print('Option2 score = %s' % option2_score)
				print('Option3 score = %s' % option3_score)
				print('Option4 score = %s' % option4_score)
				print('Option5 score = %s' % option5_score)
		
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
