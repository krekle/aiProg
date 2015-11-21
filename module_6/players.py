#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

import numpy as np

from gamelogic.board import Game, Direction
from network import ANN
from module_6.preprocessing import Process

__author__ = 'krekle'


class Player:
    """
    Abstract player class
    """
    directions = [Direction.Left, Direction.Right, Direction.Up, Direction.Down]
    games = []
    scores = []

    def __init__(self, games_count=50):
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
        random.shuffle(self.directions)

        moved = False
        tries = -1

        while moved is not True:
            tries += 1

            # Check if no possible moves => Game over
            if tries > len(self.directions) - 1:
                break  # the loop

            # As long as moves are possible continue
            else:
                moved = game.move(self.directions[tries])

        # If True -> board moved, continue
        # If False -> no possible moves, game over
        return moved


class Neural(Player):
    neural_net = None

    def __init__(self, games_count=10):
        # Load training data
        data_2048 = np.loadtxt('log-2048.txt', dtype=float, usecols=range(17))

        # Get the labels
        raw_labels_2048 = data_2048[:, 16]
        labels_2048 = [[0 for y in range(4)] for x in range(len(raw_labels_2048))]
        for i in range(len(raw_labels_2048)):
            labels_2048[i][int(raw_labels_2048[i])] = 1.0
        labels_2048 = np.array(labels_2048)

        # Get the states
        states_2048 = np.delete(data_2048, np.s_[-1:], 1)

        data = [Process.mergable_neighbours(states_2048, nested=False, shape=(4, 4)), labels_2048, states_2048,
                labels_2048]



        # Initialize neural network
        self.neural_net = ANN(nodes=(16, 8, 6, 4), data=data)

        # Train
        self.train()

        # Call super
        super(Neural, self).__init__(games_count=games_count)

    def train(self):
        """
        Method for training the network
        """
        self.neural_net.training(batch=120, verbose_level=2, epochs=10)

    def do_move(self, game):
        moves = False
        print(game.grid)
        print(self.neural_net.blind_test(np.array(game.grid)))
        pass


        # Preprocessing Methods


ran = Random()
ann = Neural()
print(ran.get_scores())
print(ann.get_scores())
