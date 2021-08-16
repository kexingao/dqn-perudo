import random
from player import Player
from bet import DUDO, create_bet
from die import Die

import copy
import os
import json

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.optim import Adam


### parameters
static_param_set = {
	'reward_decay': 0.9,
	'e_greedy': 0.2,
	'epsilon_increment': 0.001,
	'epsilon_max': 0.95,
	'learning_rate': 0.001,
	'memory_size': 100000,
	'hidden_dim': 64,
	'hidden_layer': 2,
	'target_update_interval': 200,
	'batch_size': 32,
	'start_train': 10
}

### dqn network
class NN(nn.Module):
	def __init__(self, param_set):
		super(NN, self).__init__()
		self.param_set = param_set

		self.input_len = param_set['obs_shape']
		self.output_len = param_set['n_actions']
		self.hidden_dim = param_set['hidden_dim']
		self.hidden_layer = param_set['hidden_layer']

		self.encode = nn.Linear(self.input_len, self.hidden_dim)
		self.fc = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.hidden_layer)])
		self.decode = nn.Linear(self.hidden_dim, self.output_len)

	def forward(self, inputs):

		x = F.leaky_relu(self.encode(inputs))
		for i in range(self.hidden_layer):
			x = F.leaky_relu(self.fc[i](x))
		x = self.decode(x)

		if len(inputs.shape)==1:
			x = x.reshape(-1)

		return x

class DQNPlayer(Player):
	def __init__(self, name, dice_number, game, n_player, prix=''):
		super(DQNPlayer, self).__init__(name, dice_number, game)
		self.dice_number = dice_number
		self.n_player = n_player
		self.action_set = ['Dudo']
		self.limit = dice_number * n_player // 2 + 1
		for i in range(1,self.limit):
			self.action_set += [(i,j) for j in range(1,7)]

		new_param_set = {
			'obs_shape': dice_number+4,
			'n_actions': len(self.action_set),
		}

		static_param_set.update(new_param_set)
		param_set = static_param_set

		self.n_features = param_set['obs_shape']

		self.lr = param_set['learning_rate']
		self.gamma = param_set['reward_decay']
		self.epsilon = param_set['e_greedy']
		self.epsilon_increment = param_set['epsilon_increment']
		self.epsilon_max = param_set['epsilon_max']
		self.start_train = param_set['start_train']

		self.memory = np.zeros((param_set['memory_size'], param_set['obs_shape'] * 2 + 3))  # state——id， next—state-id，action，reward
		self.priority = np.zeros((param_set['memory_size']))  # state——id， next—state-id，action，reward

		self.Q = NN(param_set)
		self.target_Q = copy.deepcopy(self.Q)
		self.params = self.Q.parameters()
		self.optimiser = Adam(params=self.params, lr=self.lr)

		self.update_frequncy = param_set['target_update_interval']
		self.last_update = 0
		self.step = 0
		self.max_memory_id = param_set['memory_size']
		self.batch_size = param_set['batch_size']
		self.memory_counter = 0
		self.state = None
		self.learn_step_counter = 0
		self.prix = prix
		self.win = 0
		self.round_step = 0
		self.round_reward = 0
		self.prob_alpha = 0.6

		self.record = []

	def reset(self):
		self.palifico_round = -1
		self.dice = []
		for i in range(0, self.dice_number):
			self.dice.append(Die())

		if len(self.memory) > 0:
			self.record.append((self.round_reward, self.round_step, self.win ))

		self.round_step = 0
		self.round_reward = 0
		self.win = 0



	def get_available_bet(self, last_bet):
		avail = [1]*len(self.action_set)

		if not last_bet:
			avail[0] = 0
			return [0], avail


		not_avail = []
		p = 1

		for quantity in range(1, self.limit):
			for value in range(1, 7):
				if self.game.is_palifico_round() and self.palifico_round == -1 and value != last_bet.value:
					not_avail.append(p)
					avail[p] = 0
				elif last_bet.value == 1 and value > 1 and quantity < last_bet.quantity * 2 + 1:
					not_avail.append(p)
					avail[p] = 0

				elif quantity <= last_bet.quantity and (
						value == last_bet.value or (value > 1 and value <= last_bet.value)):
					not_avail.append(p)
					avail[p] = 0

				elif value == 1 and last_bet.value > 1 and quantity < last_bet.quantity / 2:
					not_avail.append(p)
					avail[p] = 0

				p += 1
		return not_avail, avail

	def approximate_Q(self, state):

		state = th.FloatTensor(state)
		q = self.Q(state)
		return q

	def make_bet(self, current_bet):
		self.round_step += 1

		dices = [die.value for die in self.dice]

		if current_bet:
			current = [current_bet.quantity, current_bet.value]
		else:
			current = [-1] * 2

		count_dice = [len(player.dice) for player in self.game.players]

		self.state = dices + [-1] * (self.dice_number - len(self.dice)) + current + [sum(count_dice), len(self.dice)]

		not_avail, avail = self.get_available_bet(current_bet)
		q_value = self.approximate_Q(self.state)
		q_value[not_avail] = float('-inf')
		# print([(a, q) for a, q in zip(self.action_set,q_value)])

		if random.random() < self.epsilon:
			# print([(a, av, q) for a, av, q in zip(self.action_set, avail, q_value)])
			self.action_index = q_value.argmax()
		else:
			# probs = th.exp(q_value)
			# q_ = q_value
			# # print(probs)
			#
			# if sum(probs) < 0.01 or sum(probs) > 10000:
			# 	q_ = q_ / 10
			# 	probs = th.exp(q_)
			#
			# probs = (probs/sum(probs)).detach().numpy()
			# print(probs)
			probs = np.array(avail)/sum(avail)
			self.action_index = np.random.choice(len(self.action_set), p=probs)

		action = self.action_set[self.action_index]
		# print(action)


		try:
			bet = create_bet(action[0], action[1], current_bet, self, self.game)
			return bet
		except:
			return DUDO

	def store_transition(self, current_bet, R, done=0):
		self.round_reward += R

		# transition = np.hstack((s, [a, r], s_))
		# print(self.state, self.action_index, R)

		dices = [die.value for die in self.dice]

		count_dice = [len(player.dice) for player in self.game.players]
		if current_bet:
			current = [current_bet.quantity, current_bet.value]
		else:
			current = [0] * 2

		exp_state = dices + [-1] * (self.dice_number - len(self.dice)) + current + [sum(count_dice), len(self.dice)]


		transition = np.concatenate([self.state, [self.action_index, R, done], exp_state])  # hatack:水平（按列）顺序堆叠数组。

		# if self.memory_counter exceed memory size，so overlap original experience
		index = self.memory_counter % self.max_memory_id
		self.memory[index, :] = transition

		if self.memory_counter > 0:
			max_prior = max(self.priority)
		else:
			max_prior = 1.0
		self.priority[index] = max_prior

		self.memory_counter += 1

		if self.memory_counter > self.start_train:
			self.train()

	def train(self):

		# select batchsz sample
		probs = self.priority ** self.prob_alpha
		probs = probs / sum(probs)
		sample_index = np.random.choice(self.max_memory_id, self.batch_size, p=probs)

		# sample_index = np.random.choice(min(self.max_memory_id, self.memory_counter), size=self.batch_size)
		weights = (min(self.max_memory_id, self.memory_counter) * probs[sample_index]) ** (-0.4)
		weights = weights / weights.max()
		weights = th.FloatTensor(weights).unsqueeze(-1)

		batch_memory = self.memory[sample_index]

		obs = batch_memory[:, :self.n_features]
		next_state = th.FloatTensor(batch_memory[:, -self.n_features:])
		done = th.FloatTensor(batch_memory[:, self.n_features+2])

		q_value = self.approximate_Q(obs)

		action = th.LongTensor(batch_memory[:, self.n_features]).unsqueeze(-1)
		chosen_q = th.gather(q_value, dim=1, index=action)
		# target_q, _ = th.max(self.target_Q(next_state), dim=-1, keepdim=True)
		# target_q = (1-done) * target_q * self.gamma + th.FloatTensor(batch_memory[:, self.n_features + 1])
		#
		# loss = ((chosen_q - target_q.detach()) ** 2).mean()
		score = th.FloatTensor(batch_memory[:, self.n_features + 1]).unsqueeze(-1)
		# print(chosen_q.shape, score.shape)
		td_error = (((chosen_q - score) ** 2) * weights)
		# print(sample_index, td_error)
		for batch_index, priority in zip(sample_index, td_error):
			self.priority[batch_index] = priority

		loss = td_error.mean()

		self.optimiser.zero_grad()
		loss.backward()
		grad_norm = th.nn.utils.clip_grad_norm_(self.params, 10)
		self.optimiser.step()


		# increasing epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

		self.learn_step_counter += 1

		# if self.learn_step_counter - self.last_update > self.update_frequncy:
		# 	self.last_update = self.learn_step_counter
		# 	self.target_Q.load_state_dict(self.Q.state_dict())


		if self.learn_step_counter % 500 == 0:
			print("step:", self.learn_step_counter, "loss:", float(loss))
			self.save_model()


	def save_model(self, fn=None):
		if not os.path.exists('./checkpoint'):
			os.mkdir('./checkpoint')
		if fn:
			th.save(self.Q.state_dict(), './checkpoint/' + fn)
		else:
			th.save(self.Q.state_dict(), './checkpoint/{}mainQN_{}_{}.pth'.format(self.prix, self.n_player, self.dice_number))
			jsObj = json.dumps(self.record)

			fileObject = open('./checkpoint/{}reward_{}_{}.json'.format(self.prix, self.n_player, self.dice_number), 'w')
			fileObject.write(jsObj)
			fileObject.close()

	def load_model(self, path=None):
		if path:
			self.Q.load_state_dict(th.load(path, map_location=th.device('cpu')))
			self.target_Q.load_state_dict(th.load(path, map_location=th.device('cpu')))
		else:
			if os.path.exists('./checkpoint/mainQN_{}_{}.pth'.format(self.n_player, self.dice_number)):
				self.Q.load_state_dict(th.load('./checkpoint/mainQN_{}_{}.pth'.format(self.n_player, self.dice_number), map_location=th.device('cpu')))
				self.target_Q.load_state_dict(th.load('./checkpoint/mainQN_{}_{}.pth'.format(self.n_player, self.dice_number), map_location=th.device('cpu')))
