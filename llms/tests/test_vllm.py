#! /usr/bin/env python

from torchvision.transforms.v2.functional import hflip
from torch.nn.functional import softmax
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
	
	print('Setting up HF models')
	# tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir, use_fast=False)
	# hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="cuda", cache_dir=cache_dir)
	
	rng_gen = default_rng(40)
	train_idxs = rng_gen.choice(train_samples.shape[0], 5, replace=False)
	student_samples = [train_samples.iloc[idx].to_dict() for idx in train_idxs]
	question = test_samples.iloc[rng_gen.choice(test_samples.shape[0])].to_dict()
	context = cot_context(question, student_samples)
	print(question['question'])
	print(context)
	
	print('Making inference with vLLM')
	outputs = vllm_model.generate(context, gen_params)
	answer = outputs[0].outputs[0].text
	logprobs = outputs[0].outputs[0].logprobs
	answer_end = answer.index('\n')
	logprobs_text = [logprob.values()[0].decoded_token.strip() for logprob in logprobs]
	logprobs_vals = [logprob.values()[0].logprob for logprob in logprobs]
	answer_logprobs = logprobs_text.index('yes') if 'yes' in logprobs_text else logprobs_text.index('no')
	answer = answer[:answer_end]
	print(answer)
	print(logprobs_text[:answer_logprobs+1])
	print(logprobs_vals[:answer_logprobs+1])
	# print(softmax(logprobs[answer_logprobs][list(logprobs[answer_logprobs].keys())[0]].logprob, dim=-1))
	
	# print('Making inference with HF')
	# tokens = tokenizer([context], return_tensors="pt").to("cuda")
	# print('Generating answer')
	# generated = hf_model.generate(**tokens, num_beams=num_beams, max_new_tokens=max_tokens, output_scores=True, return_dict_in_generate=True)
	# print('Decoding answer')
	# output = tokenizer.batch_decode(generated['sequences'], skip_special_tokens=True)[0].strip()
	# output = output[len(context):]
	# output = output[:output.index('\n')]
	# output = output[0].strip()[:output.index('\n')].strip() if '\n' in output else output[0].strip().strip()
	# softmax(generated['scores'][answer_id], dim=-1)
	# print(output)
	# print(generated[0].squeeze().tolist())
	

if __name__ == '__main__':
	main()
