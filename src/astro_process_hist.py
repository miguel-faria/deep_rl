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


history_dir = Path(__file__).parent.absolute().parent.absolute() / 'models' / 'astro_disposal_dqn'
history_file = history_dir / (LEVEL + '.json')
process_history_file = history_dir / (LEVEL + '_proccessed.csv')
processed_history = []
header = ['state', 'human_action', 'robot_action']

with open(history_file, 'r') as f:
	history = json.load(f)

with open(process_history_file, 'w') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(header)
	for entry in history:
		state = entry[0]
		human_act = convert_action(int(entry[1]))
		robot_act = convert_action(int(entry[3]))
		writer.writerow([state, human_act, robot_act])

