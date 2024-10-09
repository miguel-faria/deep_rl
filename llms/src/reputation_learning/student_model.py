#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from typing import Dict, List, Union, Tuple
from reputation_learning.model import Model, UnidentifiedExplanationError, UnidentifiedTaskError
from pandas import DataFrame


class StudentModel(Model):
	
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
			print("Partial explanation = %s" % test_sample_partial_explanation)
			context += "\n\nQ: %s\nA: %s" % (test_sample['question'], test_sample_partial_explanation)
		
		else:
			raise UnidentifiedTaskError("Task %s not defined for teacher explanation context" % self._task)
		
		return context
	
	def get_context(self, sample: Dict, explanation: Union[List, str] = None, intervene: bool = False):
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
		context = self.get_context(sample, explanation=explanation)
		tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
		generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens, output_scores=True, return_dict_in_generate=True)
		output = self.tokenizer.batch_decode(generated[0], skip_special_tokens=True)[0].strip()
		
		idx = 1 if "llama" in self._model_name else 0
		if self._task == "strategy_qa":
			yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
			answer_id = 0
			
			if with_explanation and not explanation:
				if "llama" in self._model_name:
					end_id = self.tokenizer.encode("\n")[2]
					answer_id = len(tokens["input_ids"][0])
				else:
					end_id = self.tokenizer.encode("\n")[0]
					answer_id = 1
				
				generated_tokens = generated[0].squeeze().tolist()[answer_id:]
				if end_id in generated_tokens:
					generated_tokens = generated_tokens[:generated_tokens.index(end_id)]
				
				if yes_id in generated_tokens or no_id in generated_tokens:
					answer_id = generated_tokens.index(yes_id) if yes_id in generated_tokens else generated_tokens.index(no_id)
			
			scores = softmax(generated['scores'][answer_id], dim=-1)
			yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
			class_scores = [yes_score, no_score]
			
			if debug:
				print('Yes score = %s' % yes_score)
				print('No score = %s' % no_score)
		
		elif self._task == "ec_qa":
			option1_id, option2_id, option3_id, option4_id, option5_id = (self.tokenizer.encode("1")[idx], self.tokenizer.encode("2")[idx], self.tokenizer.encode("3")[idx],
																		  self.tokenizer.encode("4")[idx], self.tokenizer.encode("5")[idx])
			option1_text_id, option2_text_id, option3_text_id, option4_text_id, option5_text_id = (self.tokenizer.encode(sample["options"][0].split(" ")[0])[idx],
																								   self.tokenizer.encode(sample["options"][1].split(" ")[0])[idx],
																								   self.tokenizer.encode(sample["options"][2].split(" ")[0])[idx],
																								   self.tokenizer.encode(sample["options"][3].split(" ")[0])[idx],
																								   self.tokenizer.encode(sample["options"][4].split(" ")[0])[idx])
			
			found_text = False
			if with_explanation and not explanation:
				if "llama" in self._model_name:
					end_id = self.tokenizer.encode("\n")[2]
					answer_id = len(tokens["input_ids"][0])
				else:
					end_id = self.tokenizer.encode("\n")[0]
					answer_id = 1
				
				generated_tokens = generated[0].squeeze().tolist()[answer_id:]
				if end_id in generated_tokens:
					generated_tokens = generated_tokens[:generated_tokens.index(end_id)]
				
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
			else:
				answer_id = 0
				if output.split(" ")[0] not in ["1", "2", "3", "4", "5"]:
					found_text = True
			
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
	
	def predict(self, sample: Dict, expl: Union[List, str] = None, intervene: bool = False, debug: bool = False):
		context = self.get_context(sample=sample, explanation=expl, intervene=intervene)
		tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
		generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens)
		output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
		
		if "llama" in self._model_name:
			output = output[len(context):]
		output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
		
		if self._task == "ec_qa" and "The correct choice is " in output:
			output = output[len("The correct choice is "):].strip()
		
		if not self._use_explanations or (self._explanation_type.find("cot") == -1 and (self._explanation_type.find("chain") == -1 and self._explanation_type.find("thought") == -1)):
			if self._task == "ec_qa":
				if output not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(sample["options"]):
						if choice in output:
							output = str(i + 1)
							break
			prediction = output.split(" ")[0]
			explanation = " ".join(output.split(" ")[2:])
			if debug:
				print('Student Prediction = %s' % prediction)
				print('Student Explanation = %s' % explanation)
		else:
			explanation = output[:output.rfind(".") + 1] if self._task != "gsm8k" else output
			if debug:
				print('Student Explanation = %s' % explanation)
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
					context = self.teacher_explanation_context(sample, explanation)
					tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
					generated = self.gen_model.generate(**tokens, num_beams=self._num_beams, max_new_tokens=self._max_tokens)
					output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
					output = output[len(context):] if context in output else output
					output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
					prediction = output.split(" ")[0]
			
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
	
	def predict_batch(self, samples: DataFrame, intervention_indexes_per_budget: List[List[int]], teacher: Model, debug: bool = False) -> Tuple[List, List, List]:
		labels = []
		predictions_per_budget = [[] for _ in range(len(intervention_indexes_per_budget))]
		explanations_per_budget = [[] for _ in range(len(intervention_indexes_per_budget))]
		
		for test_index, test_sample in samples.iterrows():
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
