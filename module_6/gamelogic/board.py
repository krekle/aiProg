#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from random import randrange

__author__ = 'krekle'


class Direction:
    Up = (1, 0)
    Down = (-1, 0)
    Left = (0, -1)
    Right = (0, 1)

    @staticmethod
    def get(direction):
        if direction == 'left':
            return Direction.Left, 'left'
        elif direction == 'right':
            return Direction.Right, 'right'
        elif direction == 'up':
            return Direction.Up, 'up'
        else:
            return Direction.Down, 'down'


class Game():
    grid = None
    y_size = 0
    x_size = 0

    def __init__(self, grid=None):
        # Generate new empty grid
        if not grid:
            self.grid = [[' ' for x in range(4)] for y in range(4)]
        else:
            self.grid = grid

        self.y_size = len(self.grid)
        self.x_size = len(self.grid[0])
        # Set first value
        self.place_random()

    def generate_random(self):
        ran = randrange(0, 10)
        tile = 0
        if 0 <= ran < 8:
            tile = 2
        else:
            tile = 4
        return tile

    def place_random(self):
        # Generate a new tile value
        tile = self.generate_random()

        # Find empty tiles
        ran = []
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                if self.grid[y][x] == ' ':
                    ran.append((y, x))

        # Pick one tile at random
        if len(ran) >= 1:
            spot = ran[randrange(0, len(ran))]
        else:
            # Game ended ?
            return

        # Set the new value
        self.grid[spot[0]][spot[1]] = tile

    def pprint(self):
        for line in self.grid:
            print(line)

    def get(self, y, x):
        return self.grid[y][x]

    def set(self, y, x, value=None):
        if value is None:
            self.grid[y][x] = ' '
        else:
            self.grid[y][x] = value

    def move_line(self, line):
        # Length of original line
        n = len(line)

        # Copy original
        res = line

        # Remove all empty elements
        if ' ' in line:
            while (' ' in res):
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

    def move(self, direction):
        # For validating move
        backup_board = copy.deepcopy(self.grid)

        # First Flip if Down or Right
        if direction == Direction.Down:
            # reverse whole and use UP logic
            self.grid = list(reversed(self.grid))

        if direction == Direction.Right:
            # flip each y to use LEFT logic
            for y in range(0, self.y_size):
                self.grid[y] = list(reversed(self.grid[y]))

        # Vertical Movement
        if direction == Direction.Up or direction == Direction.Down:
            for x in range(0, self.x_size):
                # Copy current y-line
                current_line = []
                for y in range(0, self.y_size):
                    current_line.append(self.get(y, x))

                # Y-line after -> add to grid
                moved_line = self.move_line(current_line)
                for new_y in range(0, self.y_size):
                    self.set(new_y, x, moved_line[new_y])

        # Horizontal Movement
        if direction == Direction.Left or direction == Direction.Right:
            for y in range(0, self.y_size):
                # Set moved as new line
                self.grid[y] = self.move_line(self.grid[y])


        # Flip back again if Down or Right
        if direction == Direction.Down:
            # reverse whole and use UP logic
            self.grid = list(reversed(self.grid))

        if direction == Direction.Right:
            # flip each y to use LEFT logic
            for y in range(0, self.y_size):
                self.grid[y] = list(reversed(self.grid[y]))

        # If board is same before and after move, this move is illeagal
        if self.grid == backup_board:
            # TODO Check if board is full and game is over
            self.grid = backup_board
            return False
        else:
            # Place new random tile after successful move
            self.place_random()
            return True

    def calculate_score(self, external_board=None):
        board = None
        if external_board:
            board = external_board
        else:
            board = self.grid
        res = 0
        for l in board:
            r = copy.deepcopy(l)
            while ' ' in r:
                r.remove(' ')

            res += sum(r)
        return res

    def highest_tile(self):
        highest = 0
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                if self.grid[y][x] != ' ':
                    if self.grid[y][x] > highest:
                        highest = self.grid[y][x]
        return highest
