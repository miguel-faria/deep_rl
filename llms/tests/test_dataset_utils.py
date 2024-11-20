#! /usr/bin/env python

import argparse

from utilities.dataset_tasks_utils import StrategyQA, GSM8k, ECQA
from pathlib import Path


def main():
	
	parser = argparse.ArgumentParser(description='Test script for custom-made dataset utils')
	parser.add_argument('--dataset', choices=['strategy_qa', 'gsm8k', 'ec_qa'], default='strategy_qa')
	parser.add_argument('--data-dir', dest='data_dir', required=True, type=str)
	parser.add_argument('--train-filename', dest='train_filename', default='', type=str, help='Filename of the training data')
	parser.add_argument('--test-filename', dest='test_filename', default='', type=str, help='Filename of the testing data')
	parser.add_argument('--val-filename', dest='val_filename', default='', type=str, help='Filename of the validation data')
	
	args = parser.parse_args()
	if args.dataset == "strategy_qa":
		task_dataset = StrategyQA(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	elif args.dataset == "ec_qa":
		task_dataset = ECQA(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
	else:
		task_dataset = GSM8k(data_dir=Path(args.data_dir), train_filename=args.train_filename, test_filename=args.test_filename, validation_filename=args.val_filename)
		
	train_data = task_dataset.get_train_samples()
	test_data = task_dataset.get_test_samples() if args.dataset != 'strategy_qa' else task_dataset.get_validation_samples()
	# val_data = task_dataset.get_validation_samples()
	
	for row in train_data.iterrows():
		print('Row idx: %d' % row[0])
		print(row[1])
	
	
if __name__ == '__main__':
	main()
