#! /usr/bin/env python

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax
import gymnasium
from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification


# # Generate trajectories using LLM
# def generate_trajectories(llm_model, llm_tokenizer, env, num_trajectories=100, max_steps=100):
#     trajectories = []
#     for _ in range(num_trajectories):
#         observation = env.reset()
#         trajectory = {'states': [], 'actions': []}
#         for _ in range(max_steps):
#             state_text = f"Cart position: {observation[0]}, Cart velocity: {observation[1]}, Pole angle: {observation[2]}, Pole velocity: {observation[3]}"
#             input_ids = llm_tokenizer.encode(state_text, return_tensors="np")
#             output = llm_model(input_ids=input_ids)
#             action_logits = output.logits[0, -1]
#             action_probs = jax.nn.softmax(action_logits)
#             action = jax.random.choice(jax.random.PRNGKey(0), jnp.arange(env.action_space.n), p=action_probs)
#             trajectory['states'].append(observation)
#             trajectory['actions'].append(action)
#             observation, _, done, _ = env.step(action)
#             if done:
#                 break
#         trajectories.append(trajectory)
#     return trajectories
#
# # Define Policy Network
# class PolicyNetwork(nn.Module):
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(env.action_space.n)(x)
#         return x
#
# # Define distillation loss
# @jax.jit
# def distillation_loss(policy_output, llm_output):
#     return optax.softmax_cross_entropy(policy_output, llm_output).mean()
#
# # Train Policy Network
# policy_network = PolicyNetwork()
# optimizer = optax.adam(learning_rate=1e-3)
# trajectories = generate_trajectories(llm_model, llm_tokenizer, env)
# num_epochs = 10
# for epoch in range(num_epochs):
#     total_loss = 0
#     for trajectory in trajectories:
#         states = jnp.array(trajectory['states'], dtype=jnp.float32)
#         actions = jnp.array(trajectory['actions'], dtype=jnp.int32)
#
# 		optimizer, loss = jax.value_and_grad(distillation_loss)(optimizer, policy_network(states),
# 																llm_model(llm_tokenizer.batch_encode_plus(llm_prompts, return_tensors='np')['input_ids'])[0, -1])
#         total_loss += loss
#     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(trajectories)}")
#
#
# class FlaxModel(nn.Module):
# 	model: FlaxAutoModelForSequenceClassification


class PolicyNetwork(nn.Module):
	hidden_dim: int
	n_hidden_layers: int
	n_outputs: int
	
	@nn.compact
	def __call__(self, x):
		for _ in range(self.n_hidden_layers):
			x = nn.Dense(units=self.hidden_dim)(x)
			x = nn.relu(x)
		a = nn.Dense(self.n_outputs)(x)
		v = nn.Dense(1)(x)
		return v + (a - a.mean())


def main():
	model = "google-bert/bert-base-uncased"
	tokenizer = AutoTokenizer.from_pretrained(model)
	hf_model = FlaxAutoModelForSequenceClassification.from_pretrained(model)
	
	env = gymnasium.make('CartPole-v1', render_mode='human')  # Example environment, replace with your choice
	obs, *_ = env.reset()
	rng_key = jax.random.PRNGKey(0)
	env.render()
	
	for i in range(100):
		
		llm_prompt = f"Cart position: {obs[0]}, Cart velocity: {obs[1]}, Pole angle: {obs[2]}, Pole velocity: {obs[3]}"
		llm_tokens = tokenizer.encode(llm_prompt, return_tensors='jax')
		llm_output = hf_model(input_ids=llm_tokens)
		logits = llm_output.logits[-1]
		action_probs = jax.nn.softmax(logits)
		rng_key, subkey = jax.random.split(rng_key)
		action = jax.random.choice(subkey, jnp.arange(env.action_space.n), p=action_probs)
		print(action)
		obs, _, done, *_ = env.step(np.array(action))
		env.render()
		if done:
			break
	

if __name__ == '__main__':
	main()
