#! /usr/bin/env python

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax
import gymnasium
import argparse

from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification
from typing import List


def load_teacher_models(n_teachers: int, model_paths: str) -> List[]


def main( ):
	
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--n-teachers', dest='n_teachers', type=int, required=True, help='Number of teachers models to use')
	parser.add_argument('--teacher-models', dest='teachers', type=str, nargs='+', required=True, help='List of paths for the teacher models')
	parser.add_argument('--student-model', dest='student', type=str, required=True, help='Path to student model')
	parser.add_argument('--dataset', dest='dataset', default='strategyQA', type=str, help='Dataset to train student')
	parser.add_argument('--dataset_dir', dest='dataset_dir', default='', type=str, help='Path to dataset')
	parser.add_argument('--results_file', dest='result_file', default='', type=str, help='Path to results file')
	parser.add_argument('--max_new_tokens_sm', dest='max_tokens_student', default=100, type=int)
	parser.add_argument('--max_new_tokens_tm', dest='max_tokens_teacher', default=100, type=int)
	parser.add_argument('--cache_dir', dest='cache_dir', default='', type=str)
	parser.add_argument('--num_beams', dest='n_beams', default=1, type=int)
	parser.add_argument('--ic_num', dest='n_ics', default=4, type=int)
	
	
	


if __name__ == '__main__':
	main()