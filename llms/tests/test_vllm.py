#! /usr/bin/env python

from torchvision.transforms.v2.functional import hflip
from torch.nn.functional import softmax
from torch import Tensor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from utilities.dataset_tasks_utils import StrategyQA, ECQA
from typing import Tuple, List, Optional, Dict
from numpy.random import default_rng, Generator
from pathlib import Path


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

	model = 'google/gemma-2b'
	num_beams = 4
	max_tokens = 100
	# data_dir = './data/datasets/strategyqa'
	data_dir = './data/datasets/ecqa'
	cache_dir = './cache'
	# train_filename = 'train.json'
	train_filename = 'data_train.csv'
	# test_filename = 'test.json'
	test_filename = 'data_test.csv'
	# val_filename = 'validation.json'
	val_filename = 'data_val.csv'
	
	task_dataset = ECQA(data_dir=Path(data_dir), train_filename=train_filename, test_filename=test_filename, validation_filename=val_filename)
	# task_dataset = StrategyQA(data_dir=Path(data_dir), train_filename=train_filename, test_filename=test_filename, validation_filename=val_filename)
	test_samples = task_dataset.get_validation_samples()
	train_samples = task_dataset.get_train_samples()
	
	gen_params = SamplingParams(
				temperature=0.0,
				top_k=num_beams,
				max_tokens=max_tokens,
				logprobs=1,
				spaces_between_special_tokens=False,
	)
	
	print('Setting up vLLM model')
	vllm_model = LLM(model=model, trust_remote_code=True, gpu_memory_utilization=1.0)
	
	rng_gen = default_rng(40)
	train_idxs = rng_gen.choice(train_samples.shape[0], 5, replace=False)
	student_samples = [train_samples.iloc[idx].to_dict() for idx in train_idxs]
	question = test_samples.iloc[rng_gen.choice(test_samples.shape[0])].to_dict()
	context = cot_context(question, student_samples, 'ec_qa')
	# print(question['question'])
	# print(question['options'])
	# print(context)
	
	print('Making inference with vLLM')
	outputs = vllm_model.generate(context, gen_params)
	text = outputs[0].outputs[0].text
	text = text[:text.index('\n')]
	print(context)
	print(text)
	# no_id = vllm_model.get_tokenizer().encode(' no')
	# yes_id = vllm_model.get_tokenizer().encode(' yes')
	# print(no_id, yes_id)
	# nl_id = vllm_model.get_tokenizer().encode('\n')[1]
	# nldouble_id = vllm_model.get_tokenizer().encode('\n\n')[1]
	# logprobs = outputs[0].outputs[0].logprobs
	# tokens = outputs[0].outputs[0].token_ids
	
	# option1_id = vllm_model.get_tokenizer().encode('1')
	# option2_id = vllm_model.get_tokenizer().encode('2')
	# option3_id = vllm_model.get_tokenizer().encode('3')
	# option4_id = vllm_model.get_tokenizer().encode('4')
	# option5_id = vllm_model.get_tokenizer().encode('5')
	
	# print('Getting answer logprobs')
	# nl_pos = tokens.index(nl_id) if nl_id in tokens else max_tokens
	# nldouble_pos = tokens.index(nldouble_id) if nldouble_id in tokens else max_tokens
	# answer_end = nl_pos if nl_pos < nldouble_pos else nldouble_pos
	# tokens = tokens[:answer_end]
	# logprobs = logprobs[:answer_end]
	# print(option1_id, option2_id, option3_id, option4_id, option5_id)
	# print(tokens)
	# print(logprobs)
	# no_pos = tokens.index(no_id) if no_id in tokens else max_tokens
	# yes_pos = tokens.index(yes_id) if yes_id in tokens else max_tokens
	# answer_pos = yes_pos if yes_pos < no_pos else no_pos
	# answer_logprobs = logprobs[answer_pos]
	# print(answer_logprobs, list(answer_logprobs.keys()), Tensor([logprob.logprob for logprob in answer_logprobs.values()]))

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
	# print('Getting scores')
	# answer_id = 0
	# print(generated['scores'])
	# scores = softmax(generated['scores'][answer_id], dim=-1)
	# print(scores, scores.shape)
	# yes_score, no_score = scores[0][yes_id].item(), scores[0][no_id].item()
	# print(yes_id, no_id)
	# print(yes_score, no_score)
	

if __name__ == '__main__':
	main()
