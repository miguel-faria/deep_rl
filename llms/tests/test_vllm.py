#! /usr/bin/env python

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utilities.dataset_tasks_utils import StrategyQA, ECQA
from typing import Tuple, List, Optional, Dict
from numpy.random import default_rng, Generator
from pathlib import Path
from openai import OpenAI


def cot_context(test_sample: Dict, ic_samples: List[Dict], task: str) -> str:
	context = ''
	if task == 'strategy_qa':
		context += '\n\n'.join(["Q: %s\nA: %s So the answer is %s" % (ics['question'], ics['explanation'], ics['answer']) for ics in ic_samples])
		context += '\n\nQ: %s\nA:' % test_sample['question']
	
	elif task == 'ec_qa':
		context += "\n\n".join(
				['Q: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5:%s\nA: %s So the correct choice is %s' %
				 (ics['question'], ics['options'][0], ics['options'][1], ics['options'][2], ics['options'][3], ics['options'][4], ics['explanation'], ics['answer'])
				 for ics in ic_samples])
		context += ('\n\nQ: %s\nAnswer Choices:\nChoice 1: %s\nChoice 2: %s\nChoice 3: %s\nChoice 4: %s\nChoice 5: %s\nA:' %
					(test_sample['question'], test_sample['options'][0], test_sample['options'][1], test_sample['options'][2],
					 test_sample['options'][3], test_sample['options'][4]))
	return context


def main():

	# model = 'gpt2'
	# model = 'google/gemma-2b'
	model = 'microsoft/phi-1_5'
	# model = 'meta-llama/Llama-2-13b-hf'
	num_beams = 4
	max_tokens = 100
	n_logprobs = 5
	temperature = 0.0
	data_dir = './data/datasets/strategyqa'
	# data_dir = './data/datasets/ecqa'
	cache_dir = './cache'
	train_filename = 'train.json'
	# train_filename = 'data_train.csv'
	test_filename = 'test.json'
	# test_filename = 'data_test.csv'
	val_filename = 'validation.json'
	# val_filename = 'data_val.csv'
	
	# task_dataset = ECQA(data_dir=Path(data_dir), train_filename=train_filename, test_filename=test_filename, validation_filename=val_filename)
	task_dataset = StrategyQA(data_dir=Path(data_dir), train_filename=train_filename, test_filename=test_filename, validation_filename=val_filename)
	test_samples = task_dataset.get_validation_samples()
	train_samples = task_dataset.get_train_samples()
	
	# gen_params = SamplingParams(
	# 			temperature=0.0,
	# 			top_k=num_beams,
	# 			max_tokens=max_tokens,
	# 			logprobs=1,
	# 			spaces_between_special_tokens=False,
	# )
	
	print('Setting up vLLM model')
	tokenizer = AutoTokenizer.from_pretrained(model)
	# vllm_model = LLM(model=model, trust_remote_code=True, gpu_memory_utilization=1.0)
	
	client = OpenAI(
			api_key='token-a1b2c3',
			base_url='http://localhost:8000/v1',
	)

	rng_gen = default_rng(40)
	train_idxs = rng_gen.choice(train_samples.shape[0], 5, replace=False)
	student_samples = [train_samples.iloc[idx].to_dict() for idx in train_idxs]
	question = test_samples.iloc[rng_gen.choice(test_samples.shape[0])].to_dict()
	context = cot_context(question, student_samples, 'strategy_qa')

	print('Making inference with vLLM')
	outputs = client.chat.completions.create(
			model=model,
			messages=[{'role': 'user', 'content': context}],
			max_tokens=max_tokens,
			logprobs=(n_logprobs > 0),
			top_logprobs=n_logprobs,
			temperature=temperature,
	)
	# outputs = vllm_model.generate(context, gen_params)
	# text = outputs[0].outputs[0].text
	# text = text[:text.index('\n')]
	print(context)
	print(outputs.choices)
	# print(text)


if __name__ == '__main__':
	main()
