#! /usr/bin/env python
import csv
import json

from pathlib import Path


LEVEL = 'level_one'


def convert_action(action: int) -> str:
	
	if action == 0:
		return "Up"
	elif action == 1:
		return "Down"
	elif action == 2:
		return "Left"
	elif action == 3:
		return "Right"
	elif action == 4:
		return "Interact"
	else:
		return "Stay"


history_dir = Path(__file__).parent.absolute().parent.absolute() / 'models' / 'astro_disposal_dqn' / '20230801-180619'
history_file = history_dir / (LEVEL + '.json')
process_history_file = history_dir / (LEVEL + '_proccessed.csv')
processed_history = []
header = ['iteration', 'human_state', 'human_obj', 'human_action', 'robot_state', 'robot_obj', 'robot_action', 'objects']

with open(history_file, 'r') as f:
	history = json.load(f)

with open(process_history_file, 'w') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(header)
	it = 1
	for iteration in history:
		for entry in iteration:
			state = entry[0].split(' ')
			human_state = state[:4]
			human_obj = state[4:8]
			robot_state = state[8:12]
			robot_obj = state[12:16]
			objects = state[16:]
			human_act = convert_action(int(entry[1]))
			robot_act = convert_action(int(entry[3]))
			writer.writerow([it, human_state, human_obj, human_act, robot_state, robot_obj, robot_act, objects])
		it += 1

