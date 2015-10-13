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

            free, highest = self.state.free_tiles()
            # Run the algorithm one step
            if free <=4:
                root = TreeNode(self.state, None, 5)
            elif free <= 6:
                root = TreeNode(self.state, None, 4)
            else:
                root = TreeNode(self.state, None, 3)

            # Chose dept depentdent on free tiles
            direc, var = root.get_move()


            return direc
        except Exception, err:
            eprint(err)