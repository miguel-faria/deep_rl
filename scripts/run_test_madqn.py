#! /usr/bin/env python

import subprocess
from pathlib import Path


def main():
	
	src_dir = Path(__file__).parent.absolute().parent.absolute() / 'src'
	
	# DQN params
	N_AGENTS = 2
	N_LAYERS = 3
	BUFFER = 100
	GAMMA = 0.95
	LAYERS = (128, 128, 84)
	AGENTS = ("agent_1", "agent_2")
	
	# Train Params
	N_STEPS = 1000
	BATCH_SIZE = 128
	TRAIN_FREQ = 5
	TARGET_FREQ = 50
	ALPHA = 0.00025
	TAU = 0.0000025
	INIT_EPS = 1.0
	FINAL_EPS = 0.01
	EPS_DECAY = 0.95
	EPS_TYPE = "linear"
	WARMUP_STEPS = 100
	
	args =(" --nagents %d --nlayers %d --buffer %d --gamma %f --layer-sizes %d %d %d --agent-ids %s %s --gpu --steps %d --batch %d --train-freq %d "
		   "--target-freq %d --alpha %f --tau %f --init-eps %f --final-eps %f --eps-decay %f --eps-type %s --warmup-steps %d"
		   % (N_AGENTS, N_LAYERS, BUFFER, GAMMA, *LAYERS, *AGENTS, N_STEPS, BATCH_SIZE, TRAIN_FREQ, TARGET_FREQ,
			  ALPHA, TAU, INIT_EPS, FINAL_EPS, EPS_DECAY, EPS_TYPE, WARMUP_STEPS))
	try:
		subprocess.run(
			"python " + str(src_dir / 'test_madqn.py') + args,
			shell=True,
			check=True
		)
	
	except subprocess.CalledProcessError as e:
		print(e.output)


if __name__ == '__main__':
	main()
