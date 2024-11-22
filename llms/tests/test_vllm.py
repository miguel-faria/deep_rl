#! /usr/bin/env python

from torchvision.transforms.v2.functional import hflip
from torch.nn.functional import softmax
from torch import Tensor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities.dataset_tasks_utils import StrategyQA
from typing import Tuple, List, Optional, Dict
from numpy.random import default_rng, Generator
from pathlib import Path


def cot_context(test_sample: Dict, ic_samples: List[Dict]) -> str:
	context = ''
	context += '\n\n'.join(["Q: %s\nA: %s So the answer is %s" % (ics['question'], ics['explanation'], ics['answer']) for ics in ic_samples])
	context += '\n\nQ: %s\nA:' % test_sample['question']
	return context


def main():

	model = 'EleutherAI/pile-t5-base'
	num_beams = 4
	max_tokens = 100
	data_dir = './data/datasets/strategyqa'
	cache_dir = './cache'
	train_filename = 'train.json'
	test_filename = 'test.json'
	val_filename = 'validation.json'
	
	task_dataset = StrategyQA(data_dir=Path(data_dir), train_filename=train_filename, test_filename=test_filename, validation_filename=val_filename)
	test_samples = task_dataset.get_validation_samples()
	train_samples = task_dataset.get_train_samples()
	
	gen_params = SamplingParams(
				temperature=0.0,
				top_k=num_beams,
				max_tokens=max_tokens,
				logprobs=1
	)
	
	print('Setting up vLLM model')
	vllm_model = LLM(model=model, trust_remote_code=True, gpu_memory_utilization=1.0)
	
	rng_gen = default_rng(40)
	train_idxs = rng_gen.choice(train_samples.shape[0], 5, replace=False)
	student_samples = [train_samples.iloc[idx].to_dict() for idx in train_idxs]
	question = test_samples.iloc[rng_gen.choice(test_samples.shape[0])].to_dict()
	context = cot_context(question, student_samples)
	# print(question['question'])
	# print(context)
	
	print('Making inference with vLLM')
	outputs = vllm_model.generate(context, gen_params)
	answer = outputs[0].outputs[0].text
	logprobs = [list(logprob.values())[0] for logprob in outputs[0].outputs[0].logprobs]
	answer_end = answer.index('\n')
	logprobs_decoded = [logprob.decoded_token.strip() for logprob in logprobs]
	logprobs_values = Tensor([logprob.logprob for logprob in logprobs])
	print(logprobs_decoded)
	print(logprobs_values)
	answer_logprobs = logprobs_decoded.index('yes') if 'yes' in logprobs_decoded else logprobs_decoded.index('no')
	answer = answer[:answer_end]
	print(answer)
	print()
	print(softmax(logprobs_values, dim=-1)[:answer_logprobs])
	# print(answer_logprobs)
	# print(logprobs[answer_logprobs][list(logprobs[answer_logprobs].keys())[0]].logprob)


	# print('Setting up HF models')
	# tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, use_fast=False)
	# hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="cuda", cache_dir=cache_dir)
	
	# print('Making inference with HF')
	# tokens = tokenizer([context], return_tensors="pt").to("cuda")
	# yes_id, no_id = tokenizer.encode("yes")[0], tokenizer.encode("no")[0]
	
	# print('Generating answer')
	# generated = hf_model.generate(**tokens, num_beams=num_beams, max_new_tokens=max_tokens, output_scores=True, return_dict_in_generate=True)
	# print('Decoding answer')
	# output = tokenizer.batch_decode(generated['sequences'], skip_special_tokens=True)[0].strip()
	# output = output[len(context):]
	# output = output[:output.index('\n')]
	# print(output)
	# generated_tokens = generated[0].squeeze().tolist()
	# answer_id = 0
	# if yes_id in generated_tokens or no_id in generated_tokens:
	# 	answer_id = generated_tokens.index(yes_id) - 1 if yes_id in generated_tokens else generated_tokens.index(no_id) - 1
	# scores = softmax(generated['scores'][answer_id], dim=-1)
	# print(scores)
	# yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
	# print(yes_score, no_score)
	# print(generated[0].squeeze().tolist())
	

if __name__ == '__main__':
	main()
