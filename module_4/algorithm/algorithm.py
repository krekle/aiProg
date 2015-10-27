#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tree import TreeNode
from state import State
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

            free, highest = self.state.free_tiles()
            # Run the algorithm one step
            if free <= 1:
                root = TreeNode(self.state, None, 7)
            elif free <= 2:
                root = TreeNode(self.state, None, 6)
            elif free <= 3:
                root = TreeNode(self.state, None, 5)
            elif free <= 6:
                root = TreeNode(self.state, None, 4)
            else:
                root = TreeNode(self.state, None, 3)

            # root = TreeNode(self.state, None, 4)

            # Chose dept depentdent on free tiles
            direc, way = root.get_move()

            return direc, highest
        except Exception, err:
            eprint(err)
