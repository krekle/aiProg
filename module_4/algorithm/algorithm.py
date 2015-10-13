#!/usr/bin/env python
# -*- coding: utf-8 -*-

from state import State, TreeNode
from traceback import print_exc as eprint
__author__ = 'krekle'


class MinMax():
    def __init__(self, gui, game):
        self.gui = gui
        self.game = game

    def run(self):
        try:
            # Create a state of the initial game
            self.state = State(self.game.grid)
            # Start
            root = TreeNode(self.state, None, 3)
            direc, var = root.get_move()

            print var
            self.game.move(direc)

        except Exception, err:
            eprint(err)

    def MinMax(self):
        pass

    def MinMove(self):
        pass

    def MaxMove(self):
        pass
