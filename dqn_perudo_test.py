import config
import random
import sys
import time
from bet import Bet
from bet import DUDO
from player import ComputerPlayer
from player import HumanPlayer
from strings import correct_dudo
from strings import incorrect_dudo
from strings import INSUFFICIENT_BOTS
from strings import INSUFFICIENT_DICE
from strings import round_title
from strings import welcome_message
from strings import winner
from dqn_player import DQNPlayer

# "Burn all you love."
name, bot_number, dice_number = 'DQN', 1, 5
n_eps = 3

class Perudo(object):

	def __init__(self, name, player_number, dice_number, dqn_player:DQNPlayer):
		self.round = 0
		self.dqn_player = dqn_player
		self.dqn_player.game = self
		self.dqn_player.reset()
		self.bot_names = ['Winston', 'Luke', 'Jeff', 'Jia', 'Ben']

		self.players = [dqn_player]
		for i in range(0, player_number):
			self.players.append(
				ComputerPlayer(
					name=self.get_random_name(),
					dice_number=dice_number,
					game=self
				)
			)

		random.shuffle(self.players)

		print (welcome_message(self.players))

		self.first_player = random.choice(self.players)


		while len(self.players) > 1:
			self.run_round()
			if len(self.dqn_player.dice) == 0:
				print('AI out at round {}'.format(self.round))
				break

		if len(self.players) == 1:
			print (winner(self.players[0].name))
		else:
			print('rest player are {}'.format([p.name for p in self.players]))

	def run_round(self):
		
		ai_has_play = False
		self.round += 1
		for player in self.players:
			player.roll_dice()

		print (round_title(round_number=self.round, is_palifico_round=self.is_palifico_round()))
		round_over = False
		current_bet = None
		current_player = self.first_player
		print ('{0} will go first...'.format(current_player.name))
		while not round_over:
			if current_player == self.dqn_player:
				ai_has_play = True

			next_player = self.get_next_player(current_player)
			next_bet = current_player.make_bet(current_bet)
			bet_string = None
			if next_bet == DUDO:
				bet_string = 'Dudo!'
			else:
				bet_string = next_bet
			print ('{0}: {1}'.format(current_player.name, bet_string))
			if next_bet == DUDO:
				self.pause(0.5)
				self.run_dudo(current_player, current_bet)
				round_over = True
			else:
				current_bet = next_bet

			if len(self.players) > 1:
				current_player = next_player

			self.pause(0.5)

		self.pause(1)

	def run_dudo(self, player, bet):
		dice_count = self.count_dice(bet.value)
		if dice_count >= bet.quantity:
			print (incorrect_dudo(dice_count, bet.value))
			self.first_player = player
			self.remove_die(player)

		else:
			print (correct_dudo(dice_count, bet.value))
			previous_player = self.get_previous_player(player)
			self.first_player = previous_player
			self.remove_die(previous_player)

	def count_dice(self, value):
		number = 0
		for player in self.players:
			number += player.count_dice(value)

		return number

	def remove_die(self, player):
		player.dice.pop()
		msg = '{0} loses a die.'.format(player.name)
		if len(player.dice) == 0:
			msg += ' {0} is out!'.format(player.name)
			self.first_player = self.get_next_player(player)
			self.players.remove(player)
		elif len(player.dice) == 1 and player.palifico_round == -1:
			player.palifico_round = self.round + 1
			msg += ' Last die! {0} is palifico!'.format(player.name)
		else:
			msg += ' Only {0} left!'.format(len(player.dice))
		print (msg)

	def is_palifico_round(self):
		if len(self.players) < 3:
			return False
		for player in self.players:
			if player.palifico_round == self.round:
				return True
		return False

	def get_random_name(self):
		random.shuffle(self.bot_names)
		return self.bot_names.pop()

	def get_next_player(self, player):
		return self.players[(self.players.index(player) + 1) % len(self.players)]

	def get_previous_player(self, player):
		return self.players[(self.players.index(player) - 1) % len(self.players)]

	def pause(self, duration):
		if config.play_slow:
			time.sleep(duration)



def main():
	AI = DQNPlayer(name, dice_number, None, bot_number+1)

	AI.epsilon = 1
	AI.epsilon_max = 1
	AI.load_model()

	# AI.load_model(path='./checkpoint/DQNwin_2_3_1')
	# AI.load_model(path='./checkpoint/teacher_mainQN_{}_{}.pth'.format(bot_number+1, dice_number))
	# AI.load_model(path='./checkpoint/stu_mainQN_{}_{}.pth'.format(bot_number+1, dice_number))

	for _ in range(n_eps):
		Perudo(name, bot_number, dice_number, AI)



if __name__ == '__main__':
	main()
