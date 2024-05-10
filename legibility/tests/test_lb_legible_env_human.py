#! /usr/bin/env python
import numpy as np
import flax.linen as nn
import yaml

from dl_envs.lb_foraging.lb_foraging_coop import FoodCOOPLBForaging
from dl_envs.lb_foraging.lb_foraging import LBForagingEnv
from dl_algos.single_model_madqn import LegibleSingleMADQN
from itertools import product
from pathlib import Path
from cv2 import imwrite, cvtColor, COLOR_BGR2RGB
from typing import List, Tuple

np.set_printoptions(precision=5, threshold=10000)

ACTION_MAP = {0: 'None', 1: 'Up', 2: 'Down', 3: 'Left', 4: 'Right', 5: 'Load'}
KEY_MAP = {'w': 1, 's': 2, 'a': 3, 'd': 4, 'q': 0, 'e': 5}
RNG_SEED = 25456789
ADV_TEMP = 0.3
Q_TEMP = 20


def get_live_obs_goals(env: FoodCOOPLBForaging) -> Tuple[List, List]:
	
	live_goals = []
	goals_obs = []
	for food in env.foods:
		print(str(food.position))
		if not food.picked:
			live_goals.append(str(food.position))
			goals_obs.append(env.make_target_grid_observations(food.position))
		
	return live_goals, goals_obs


def main():
	
	n_agents = 1
	n_players = 2
	player_level = 1
	field_size = (8, 8)
	n_foods = 8
	n_foods_spawn = 2
	sight = 8
	max_steps = 5000
	food_level = 2
	img_idx = 1
	data_dir = Path(__file__).parent.absolute().parent.absolute() / 'data'
	models_dir = Path(__file__).parent.absolute().parent.absolute() / 'models'
	img_dir = data_dir / 'stills'
	Path.mkdir(img_dir, parents=True, exist_ok=True)
	with open(data_dir / 'configs' / 'lbforaging_plan_configs.yaml') as file:
		config_params = yaml.full_load(file)
		dict_idx = str(field_size[0]) + 'x' + str(field_size[1])
		if dict_idx in config_params['food_locs'].keys():
			food_locs = config_params['food_locs'][dict_idx]
			food_confs = config_params['food_confs'][dict_idx][(n_foods_spawn - 1)]
		else:
			food_locs = [tuple(x) for x in product(range(field_size[0]), range(field_size[1]))]
	optim_dir = (models_dir / 'lb_coop_single_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_players) /
				 ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / 'best')
	# optim_dir = (models_dir / 'lb_coop_legible_dqn' / ('%dx%d-field' % (field_size[0], field_size[1])) / ('%d-agents' % n_players) /
	# 			 ('%d-foods_%d-food-level' % (n_foods_spawn, food_level)) / 'best')
	optim_models = [fname.name for fname in optim_dir.iterdir()]
	
	obj_food = food_locs[0]
	env = FoodCOOPLBForaging(n_players, player_level, field_size, n_foods, sight, max_steps, True, food_level, RNG_SEED, food_locs, food_locs[1],
							 render_mode=['rgb_array', 'human'], use_encoding=False, agent_center=True, grid_observation=True)
	# env = LBForagingEnv(n_agents, player_level, field_size, n_foods, sight, max_steps, True, render_mode=['rgb_array', 'human'], grid_observation=True,
	# 					agent_center=False)
	# n_food_spawn = np.random.choice(range(n_foods))
	
	# Get optimal models
	n_layers = 2
	layer_sizes = [256, 256]
	buffer_size = 10000
	gamma = 0.95
	beta = 1.2
	use_gpu = True
	dueling_dqn = True
	use_ddqn = True
	use_cnn = True
	use_tensorboard = False
	tensorboard_details = []
	goals = [str(loc) for loc in food_locs]
	dqn_model = LegibleSingleMADQN(n_agents, env.action_space[0].n, n_layers, nn.relu, layer_sizes, buffer_size, gamma, beta, env.observation_space[0],
								   use_gpu, False, optim_dir, optim_models, goals, str(obj_food), dueling_dqn, use_ddqn, use_cnn, use_tensorboard,
								   tensorboard_details + ['%df-%dx%d-legible' % (n_foods_spawn, obj_food[0], obj_food[1])])
	
	
	# if food_confs is not None:
	# 	food_conf = food_confs[np.random.choice(range(len(food_confs)))]
	# 	locs = [food for food in food_locs if food != obj_food]
	# 	foods_spawn = [locs[idx] for idx in food_conf]
	# 	env.food_spawn_pos = foods_spawn
	rng_gen = np.random.default_rng(RNG_SEED)
	env.set_objective(obj_food)
	env.seed(seed=RNG_SEED)
	env.spawn_food(n_foods_spawn, food_level)
	env.spawn_players()
	print('Food objective is (%d, %d)' % (obj_food[0], obj_food[1]))
	print('Foods spawned: ' + str(obj_food) + ' ' +  str(env.food_spawn_pos))
	obs, *_ = env.reset(seed=RNG_SEED)
	# env.spawn_players([1, 1], [(2, 1), (5, 3)])
	print(env.get_full_env_log())
	img = env.render()
	# imwrite(str(img_dir / ('still_%d.png' % img_idx)), cvtColor(img, COLOR_BGR2RGB))
	input()
	
	for i in range(100):

		print('Iteration: %d' % (i + 1))
		actions = []
		for a_idx in range(n_players):
			if a_idx < 1:
				valid_action = False
				while not valid_action:
					human_input = input("Action for agent %d:\t" % (a_idx + 1))
					try:
						action = int(KEY_MAP[human_input])
						if action < 6:
							valid_action = True
							actions.append(action)
						else:
							print('Action ID must be between 0 and 5, you gave ID %d' % action)
					except KeyError as e:
						print('Key error caught: %s' % str(e))
			else:
				online_params = dqn_model.optimal_models[dqn_model.goal].params
				if dqn_model.agent_dqn.cnn_layer:
					q_values = dqn_model.agent_dqn.q_network.apply(online_params, obs[a_idx].reshape((1, *obs[a_idx].shape)))[0]
				else:
					q_values = dqn_model.agent_dqn.q_network.apply(online_params, obs[a_idx])
				pol = np.isclose(q_values, q_values.max(), rtol=1e-10, atol=1e-10).astype(int)
				pol = pol / pol.sum()
				action = rng_gen.choice(range(env.action_space[0].n), p=pol)
				actions.append(action)

		print('Actions: ' + ' & '.join([ACTION_MAP[action] for action in actions]))
		live_goals, goal_obs = get_live_obs_goals(env)
		next_obs, rewards, done, timeout, info = env.step(actions)
		print(env.get_full_env_log())
		
		# legible_rewards = np.zeros(dqn_model.num_agents)
		n_goals = env.food_spawn
		for a_idx in range(dqn_model.num_agents):
			adv_act_q_vals = np.zeros(n_goals)
			greedy_act_q_vals = np.zeros(n_goals)
			action = actions[a_idx]
			goal_action_q = 0.0
			for g_idx in range(n_goals):
				g_obs = goal_obs[g_idx][a_idx]
				
				if dqn_model.agent_dqn.cnn_layer:
					obs_reshape = g_obs.reshape((1, *g_obs.shape))
					q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[live_goals[g_idx]].params, obs_reshape)[0]
				else:
					q_vals = dqn_model.agent_dqn.q_network.apply(dqn_model.optimal_models[live_goals[g_idx]].params, g_obs)
				if dqn_model.goal == live_goals[g_idx]:
					goal_action_q = q_vals[action]
				# print('Goal & action index: %s %s' % (str(live_goals[g_idx]), str(action)), str(dqn_model.beta * (q_vals - q_vals.max()) / TEMP),
				# 	  str(np.exp(dqn_model.beta * (q_vals - q_vals.max()) / TEMP)))
				adv_act_q_vals[g_idx] = np.exp(dqn_model.beta * (q_vals[action] - q_vals.max()) / ADV_TEMP)
				greedy_act_q_vals[g_idx] = np.exp(dqn_model.beta * q_vals[action] / Q_TEMP)
			# logger.info("Q-vals:\t" + str(act_q_vals / act_q_vals.sum()))
			print(str(rewards))
			print('Advantage saliency')
			print(obj_food, action, str(adv_act_q_vals))
			print(str(adv_act_q_vals / adv_act_q_vals.sum()))
			print(str((adv_act_q_vals / adv_act_q_vals.sum()) * rewards[a_idx]))
			print(str((adv_act_q_vals / adv_act_q_vals.sum()) * goal_action_q))
			print(str((adv_act_q_vals / adv_act_q_vals.sum()) + rewards[a_idx]))
			# print("")
			# print('Q_val saliency')
			# print(obj_food, action, str(greedy_act_q_vals))
			# print(str(greedy_act_q_vals / greedy_act_q_vals.sum()))
			# print(str((greedy_act_q_vals / greedy_act_q_vals.sum()) * rewards[a_idx]))
			# print(str((greedy_act_q_vals / greedy_act_q_vals.sum()) * goal_action_q))
			# print(str((greedy_act_q_vals / greedy_act_q_vals.sum()) + rewards[a_idx]))
			print("")
			print('Advantage times Q_val saliency')
			print(obj_food, action, str(adv_act_q_vals * greedy_act_q_vals))
			print(str((adv_act_q_vals * greedy_act_q_vals) / (adv_act_q_vals * greedy_act_q_vals).sum()))
			print(str(((adv_act_q_vals * greedy_act_q_vals) / (adv_act_q_vals * greedy_act_q_vals).sum()) * rewards[a_idx]))
			print(str(((adv_act_q_vals * greedy_act_q_vals) / (adv_act_q_vals * greedy_act_q_vals).sum()) * goal_action_q))
			print(str(((adv_act_q_vals * greedy_act_q_vals) / (adv_act_q_vals * greedy_act_q_vals).sum()) + rewards[a_idx]))
			# legible_rewards[a_idx] = adv_act_q_vals[live_goals.index(dqn_model.goal)] / adv_act_q_vals.sum()
			# legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) * goal_action_q
			# legible_rewards[a_idx] = (act_q_vals[live_goals.index(dqn_model.goal)] / act_q_vals.sum()) * rewards[a_idx]
		# print(obj_food, actions, legible_rewards, rewards)
		
		if done or timeout:
			env.food_spawn_pos = None
			obs, *_ = env.reset()
			print(env.get_full_env_log())
		img = env.render()
		img_idx += 1
		# imwrite(str(img_dir / ('still_%d.png' % img_idx)), cvtColor(img, COLOR_BGR2RGB))
		input()
		obs = next_obs


if __name__ == '__main__':
	main()
