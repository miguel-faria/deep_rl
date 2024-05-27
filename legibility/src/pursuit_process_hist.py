#! /usr/bin/env python
import csv
import json

from pathlib import Path
from dl_envs.pursuit.pursuit_env import Action


HUNTERS = 4
PREYS = 1
FIELD_DIMS = (10, 10)
EXTRA = ''
MIN_EPS_RECORD = 2500
TIMESTAMP = ''


def convert_action(action: int) -> str:
	
	if action == Action.UP:
		return "Up"
	elif action == Action.DOWN:
		return "Down"
	elif action == Action.LEFT:
		return "Left"
	elif action == Action.RIGHT:
		return "Right"
	else:
		return "Stay"


history_dir = (Path(__file__).parent.absolute().parent.absolute() / 'models' / 'pursuit_dqn' /
			   ('%dx%d-field%s' % (FIELD_DIMS[0], FIELD_DIMS[1], EXTRA)) / ('%d-hunters' % HUNTERS)) / ('%s' % TIMESTAMP)
history_file = history_dir / ('%d-catch.json' % HUNTERS)
process_history_file = history_dir / ('%d-catch_proccessed.csv' % HUNTERS)
processed_history = []
header = ['episode'] + ['hunter_%d_pos' % (idx + 1) for idx in range(HUNTERS)] + ['prey_pos'] + ['hunter_%d_action' % (idx + 1) for idx in range(HUNTERS)]

with open(history_file, 'r') as f:
	history = json.load(f)

with open(process_history_file, 'w') as csv_file:
	writer = csv.writer(csv_file)
	writer.writerow(header)
	episode_nr = 1
	for episode in history:
		if episode_nr >= MIN_EPS_RECORD:
			for entry in episode:
				state_parse = entry[0].split(" ")
				row = ([episode_nr] + [(state_parse[idx], state_parse[idx+1]) for idx in range(0, len(state_parse), 5)] +
					   [convert_action(int(entry[idx])) for idx in range(1, len(entry), 2)])
				writer.writerow(row)
		episode_nr += 1

