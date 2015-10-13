#!/usr/bin/env python
# -*- coding: utf-8 -*-

from state import State, Node
from traceback import print_exc as eprint
__author__ = 'krekle'


class MinMax():
    def __init__(self, gui, game):
        self.gui = gui
        self.game = game

    def run(self):
        try:
            self.state = State(self.game.grid)
            root = Node(self.state, None, 3)

            for n in root.children:
                print n.choice
                print n.score

        except Exception, err:
            eprint(err)

    def MinMax(self):
        pass

    def MinMove(self):
        pass

    def MaxMove(self):
        pass
