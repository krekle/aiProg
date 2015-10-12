#!/usr/bin/env python
# -*- coding: utf-8 -*-

from state import State

__author__ = 'krekle'


class MinMax():
    def __init__(self, gui, game):
        self.gui = gui
        self.game = game


    def run(self):
        self.state = State(self.game.grid)
        self.state.generate_successors()

    def MinMax(self):
        pass

    def MinMove(self):
        pass

    def MaxMove(self):
        pass
