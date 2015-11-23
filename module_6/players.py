#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import random
from module_6.preprocessing import Process
import numpy as np

from gamelogic.board import Game, Direction
from network import ANN
from module_6.demo.ai2048demo import welch

__author__ = 'krekle'

from sys import platform as _platform

system_divider = None
if _platform == "linux" or _platform == "linux2":
    system_divider = '/'
elif _platform == "darwin":
    system_divider = '/'
elif _platform == "win32":
    system_divider = '\\'


class Player:
    """
    Abstract player class
    """
    # Static Fields
    directions = [Direction.Left, Direction.Up, Direction.Right, Direction.Down]

    def __init__(self, games_count=50):
        # Create Fields
        self.games = []
        self.scores = []

        # Create games
        for i in range(games_count):
            self.games.append(Game())

        # Play all games
        self.play_games()

    def play_games(self):
        """
        Method for playing games one after another, stores their scores
        """
        for i in range(len(self.games)):
            # Do moves until game over
            while self.do_move(self.games[i]):
                pass

            # Game over, save the score
            self.scores.append(self.games[i].highest_tile())

    # Need override in subclass
    def do_move(self):
        pass

    def get_scores(self, descending=True):
        """
        Method for returning the highest tile for each game, sorted.
        :param descending: Default sorting descending
        :return: List of all game scores
        """
        return sorted(self.scores, reverse=descending)


class Random(Player):
    def do_move(self, game):
        # Randomize moves
        dir_copy = copy.deepcopy(self.directions)
        random.shuffle(dir_copy)

        moved = False
        tries = -1

        while moved is not True:
            tries += 1

            # Check if no possible moves => Game over
            if tries > len(dir_copy) - 1:
                break  # the loop

            # As long as moves are possible continue
            else:
                moved = game.move(dir_copy[tries])

        # If True -> board moved, continue
        # If False -> no possible moves, game over
        return moved


class Neural(Player):
    neural_net = None

    def __init__(self, games_count=50, preprocess=False):
        self.is_preprocess = preprocess
        # Load training data
        data_2048 = np.loadtxt('data' + system_divider + 'log-2048-snake.bak.txt', dtype=float, usecols=range(17))

        # Get the labels
        raw_labels_2048 = data_2048[:, 16]
        labels_2048 = [[0 for y in range(4)] for x in range(len(raw_labels_2048))]
        for i in range(len(raw_labels_2048)):
            labels_2048[i][int(raw_labels_2048[i])] = 1.0

        labels_2048 = np.array(labels_2048)

        # Get the states
        states_2048 = np.delete(data_2048, np.s_[-1:], 1)

        # Preprocess training data
        if self.is_preprocess:
            states_2048 = self.preprocess(states_2048)

        data = [states_2048, labels_2048, states_2048,
                labels_2048]



        # Initialize neural network
        self.neural_net = ANN(nodes=(24, 100, 4), data=data)

        # Train
        self.train()

        # Call super
        super(Neural, self).__init__(games_count=games_count)

    def preprocess(self, data):
        shape = np.array(data).shape
        # If list of flat lists
        if shape[0] > shape[1]:
            copy = []
            for i in range(shape[0]):
                copy.append(Process.multiple_methods(argument=data[i], methods=[Process.logarithm, Process.moveable_lines]))

            return copy
        # If one single
        else:
            return Process.multiple_methods(argument=data, methods=[Process.logarithm, Process.moveable_lines])

    def train(self, batch=10, verbose_level=2, epochs=1):
        """
        Method for training the network
        """
        print('Training ...')
        self.neural_net.train(batch=batch, verbose_level=verbose_level, epochs=epochs)

    def do_move(self, game):

        prediction = None

        # Should preprocess ?
        if self.is_preprocess:
            # Preprocess to match training
            n = self.preprocess(game.grid)

            # Do move
            prediction = self.neural_net.blind_test([n])[0]
        else:
            flat = [np.array(game.grid).flatten()]
            prediction = self.neural_net.blind_test(flat)[0]

        print(prediction)
        for pred in prediction:
            if game.move(self.directions[pred]):
                return True

        # No lleagal moves, game over
        return False


def one(label):
    print('Run: ' + str(label))
    ran = Random()
    ann = Neural(preprocess=True)

    print('Ran: ' + str(ran.get_scores()))
    print('Ann: ' + str(ann.get_scores()))

    # print(np.average(ran.get_scores()))
    # print(np.average(ann.get_scores()))
    # avg.append(np.average(ann.get_scores()))

    # Welch Result
    print(welch(ran.get_scores(), ann.get_scores()))

    # Return avg ann score
    return np.average(ann.get_scores())


avg = []
for i in range(10):
    avg.append(one(i + 1))
print(np.average(avg))
