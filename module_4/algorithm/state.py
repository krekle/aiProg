#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from module4.game.board import Direction

__author__ = 'krekle'

x_size = 4
y_size = 4


class State():
    board = None
    score = None

    def __init__(self, board):
        self.board = board

    def calculate_score(self):
        pass

    def generate_successors(self):

        # Up
        up = self.simulate_move(self.board, Direction.Up)

        # Down
        down = self.simulate_move(self.board, Direction.Down)

        # Left
        left = self.simulate_move(self.board, Direction.Left)

        # Right
        right = self.simulate_move(self.board, Direction.Right)

        self.pprint(self.board)
        print 'Up'
        self.pprint(up)
        print 'Down'
        self.pprint(down)
        print 'Left'
        self.pprint(left)
        print 'Right'
        self.pprint(right)
        print ''


    def pprint(self, li):
        for l in li:
            print l

    def simulate_move(self, board, direction):
        new_board = copy.deepcopy(board)

        # First Flip if Down or Right
        if direction == Direction.Down:
            # reverse whole and use UP logic
            new_board = list(reversed(new_board))

        if direction == Direction.Right:
            # flip each y to use LEFT logic
            for y in range(0, y_size):
                new_board[y] = list(reversed(new_board[y]))

        # Vertical Movement
        if direction == Direction.Up or direction == Direction.Down:
            for x in range(0, x_size):
                # Copy current y-line
                current_line = []
                for y in range(0, y_size):
                    current_line.append(new_board[y][x])

                # Y-line after -> add to grid
                moved_line = self.move_line(current_line)
                for new_y in range(0, y_size):
                    new_board[new_y][x] = moved_line[new_y]

        # Horizontal Movement
        if direction == Direction.Left or direction == Direction.Right:
            for y in range(0, y_size):

                # Set moved as new line
                new_board[y] = self.move_line(new_board[y])


        # Flip back again if Down or Right
        if direction == Direction.Down:
            # reverse whole and use UP logic
            new_board = list(reversed(new_board))

        if direction == Direction.Right:
            # flip each y to use LEFT logic
            for y in range(0, y_size):
                new_board[y] = list(reversed(new_board[y]))

        return new_board

    def move_line(self, line):

        # Length of original line
        n = len(line)

        # Copy original
        res = line

        # Remove all empty elements
        if ' ' in line:
            while ' ' in res:
                res.remove(' ')

        # Combine and remove logic
        for i in range(0, len(res)):
            if i < len(res) and i + 1 < len(res):
                if res[i] == res[i + 1]:
                    res[i] = res[i] + res[i + 1]
                    res[i + 1] = ' '

        # Remove evt new empty
        if ' ' in res:
            res.remove(' ')


        # Fill the end with empty values
        return res + [' '] * (n - len(res))
