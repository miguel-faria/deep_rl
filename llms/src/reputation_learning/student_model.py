#! /usr/bin/env python
import re

from torch.nn.functional import softmax
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List
from model import Model


class StudentModel(Model):
	
	def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, model_name: str, expl_type: str, task: str, max_tokens: int, num_beams: int, samples: List[Dict],
	             use_explanations: bool):
		
		super().__init__(model, tokenizer, model_name, expl_type, task, max_tokens, num_beams, samples)
		self._use_explanations = use_explanations
	
	def prepare_context_teacher_explanation(self, test_sample, teacher_explanation):
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
			context = "\n\n".join([ "Q: %s\nA: %s So the answer is %s" % (sample['question'], sample['explanation'], sample['answer']) for sample in self._ic_samples])
			test_sample_explanation_sents = teacher_explanation.split(".")
			test_sample_partial_explanation = test_sample_explanation_sents[0] + "."
			print(f"Partial explanation = {test_sample_partial_explanation}")
			context += "\n\nQ: %s\nA: %s" % (test_sample['question'], test_sample_partial_explanation)
		
		else:
			assert False, "Dataset not recognized"
		
		return context
	
	def predict_single_confidence(self, test_sample, expl=None, with_expl=False):
		if not expl:
			context = self.prepare_context_no_expl(test_sample=test_sample) if not with_expl else self.prepare_context_CoT(test_sample=test_sample)
		else:
			context = self.prepare_context_teacher_explanation(test_sample=test_sample, teacher_explanation=expl)
		tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
		generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens, output_scores=True, return_dict_in_generate=True)
		output = self.tokenizer.batch_decode(generated[0], skip_special_tokens=True)[0].strip()
		
		idx = 1 if "llama" in self._model_name else 0
		if self._task == "strategy_qa":
			yes_id, no_id = self.tokenizer.encode("yes")[idx], self.tokenizer.encode("no")[idx]
			
			if with_expl and not expl:
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
				else:
					answer_id = 0
			
			else:
				answer_id = 0
			
			scores = softmax(generated['scores'][answer_id], dim=-1)
			yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
			print(f'Yes score = {yes_score}')
			print(f'No score = {no_score}')
			class_scores = [yes_score, no_score]
		elif self._task == "ec_qa":
			option1_id, option2_id, option3_id, option4_id, option5_id = (self.tokenizer.encode("1")[idx], self.tokenizer.encode("2")[idx], self.tokenizer.encode("3")[idx],
			                                                              self.tokenizer.encode("4")[idx], self.tokenizer.encode("5")[idx])
			option1_text_id, option2_text_id, option3_text_id, option4_text_id, option5_text_id = (self.tokenizer.encode(test_sample["options"][0].split(" ")[0])[idx],
			                                                                                       self.tokenizer.encode(test_sample["options"][1].split(" ")[0])[idx],
			                                                                                       self.tokenizer.encode(test_sample["options"][2].split(" ")[0])[idx],
			                                                                                       self.tokenizer.encode(test_sample["options"][3].split(" ")[0])[idx],
			                                                                                       self.tokenizer.encode(test_sample["options"][4].split(" ")[0])[idx])
			
			found_text = False
			if with_expl and not expl:
				if "llam" in self._model_name:
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
			print('Option1 score = %s' % option1_score)
			print('Option2 score = %s' % option2_score)
			print('Option3 score = %s' % option3_score)
			print('Option4 score = %s' % option4_score)
			print('Option5 score = %s' % option5_score)
			class_scores = [option1_score, option2_score, option3_score, option4_score, option5_score]
		else:
			assert False, "Dataset not recognized"
		
		return class_scores
	
	def predict_single(self, test_sample, tm, inside_mm=False, intervene=False):
		context = self.prepare_context(test_sample=test_sample, inside_mm=inside_mm, intervene=intervene, tm=tm)
		tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
		generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
		output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
		
		if "llama" in self.model_name:
			output = output[len(context):]
		output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
		
		if self.dataset == "ecqa" and "The correct choice is " in output:
			output = output[len("The correct choice is "):].strip()
		
		if not self.use_explanations or self.no_intervention_action != "CoT":
			if self.dataset == "ecqa":
				if output not in ["1", "2", "3", "4", "5"]:
					for i, choice in enumerate(test_sample["options"]):
						if choice in output:
							output = str(i + 1)
							break
			prediction = output.split(" ")[0]
			print(f'Student Prediction = {prediction}')
			explanation = " ".join(output.split(" ")[2:])
			print(f'Student Explanation = {explanation}')
		else:
			explanation = output[:output.rfind(".") + 1] if self.dataset != "gsm8k" else output
			print(f'Student Explanation = {explanation}')
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
					context = self.prepare_context_teacher_explanation(test_sample, explanation)
					tokens = self.tokenizer([context], return_tensors="pt").to("cuda")
					generated = self.model.generate(**tokens, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens)
					output = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
					output = output[len(context):] if context in output else output
					output = output[:output.index('\n')].strip() if '\n' in output else output.strip()
					prediction = output.split(" ")[0]
			
			elif self.dataset == "gsm8k":
				prediction = re.sub(r"[^0-9.]", "", prediction)
				if prediction == "" or prediction == ".":
					for word in reversed(explanation.split(" ")):
						if bool(re.search(r"\d", word)):
							prediction = re.sub(r"[^0-9.]", "", word)
							break
			
			print(f'Student Prediction = {prediction}')
		
		return prediction, explanation
	
	def predict(self, test_samples, intervention_samples_per_budget, tm):
		questions, labels, predictions_per_budget, explanations_per_budget = [], [], [[] for i in range(len(intervention_samples_per_budget))], [[] for i in range(len(intervention_samples_per_budget))]
		
		for test_index, test_sample in enumerate(tqdm(test_samples)):
			print("Using student explanation")
			prediction_student_expl, explanation_student = self.predict_single(test_sample=test_sample, tm=tm, intervene=False)
			
			print("Using teacher explanation")
			# This is not actually explanation teacher, but don't care for final student evaluation
			prediction_teacher_expl, explanation_teacher = self.predict_single(test_sample=test_sample, tm=tm, intervene=True)
			
			for i, intervention_samples in enumerate(intervention_samples_per_budget):
				if test_index in intervention_samples:
					predictions_per_budget[i].append(prediction_teacher_expl)
					explanations_per_budget[i].append(explanation_teacher)
				else:
					predictions_per_budget[i].append(prediction_student_expl)
					explanations_per_budget[i].append(explanation_student)
			
			questions.append(test_sample['question'])
			labels.append(test_sample['answer'])
		
		return questions, labels, predictions_per_budget, explanations_per_budget
