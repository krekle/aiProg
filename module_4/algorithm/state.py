#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from game.board import Direction, Game

__author__ = 'krekle'

x_size = 4
y_size = 4


class State():
    def __init__(self, board):
        self.score = None
        self.board = board
        # self.calculate_score() -> Dont calculate score for middle nodes

    def free_tiles(self):
        free = 0
        highest_tile = 2
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x] == ' ':
                    free += 1
                else:
                    # Check highest tile
                    if self.board[y][x] > highest_tile:
                        highest_tile = self.board[y][x]

        return free, highest_tile


    def calculate_score(self):

        score = 0

        # Highest til and Number of free spaces > = more flexible later on
        free, highest_tile = self.free_tiles()
        score += 30 * free
        score += highest_tile * 5

        # check number of adjecent that can be combined next time
        # High tile with only 2 neighbours = corner

        # Snake: Top left is worth the most
        snake = [60, 40, 25, 10]
        for i in range(len(snake)):
            ch = self.board[0][i]
            if str(ch).isdigit():
                score += int(ch) * snake[i]

        snake2 = [40, 25, 10, 1]
        for k in range(len(snake2)):
            ch = self.board[1][k]
            if str(ch).isdigit():
                score += int(ch) * snake2[k]

        snake3 = [25, 10, 1, 1]
        for j in range(len(snake3)):
            ch = self.board[2][j]
            if str(ch).isdigit():
                score += int(ch) * snake2[j]

        snake4 = [10, 1, 1, 1]
        for h in range(len(snake4)):
            ch = self.board[3][h]
            if str(ch).isdigit():
                score += int(ch) * snake2[h]


        # Hard penalty for lose game
        if free == 0:
            score = 0

        # Save the score
        self.score = score
        return self.score


    def max_successors(self):
        d = dict()

        # Up
        up_state = self.simulate_move(self.board, Direction.Up)
        if up_state:
            d['up'] = State(up_state)

        # Down
        down_state = self.simulate_move(self.board, Direction.Down)
        if down_state:
            d['down'] = State(down_state)

        # Left
        left_state = self.simulate_move(self.board, Direction.Left)
        if left_state:
            d['left'] = State(left_state)

        # Right
        right_state = self.simulate_move(self.board, Direction.Right)
        if right_state:
            d['right'] = State(right_state)

        return d

    def min_successors(self):
        children = []
        free_spots = []
        for y in range(y_size):
            for x in range(x_size):
                if self.board[y][x] == ' ':
                    free_spots.append((y, x))

        # Add states on the free spots, 2 and 4
        for coord in free_spots:
            # 2:
            children.append(self.add_min_state(2, coord))
            # 4:
            children.append(self.add_min_state(4, coord))

        return children

    def add_min_state(self, value, coord):
        # Method for adding a state with value at coordinate
        new_board = copy.deepcopy(self.board)
        new_board[coord[0]][coord[1]] = value
        state = State(new_board)
        return State(new_board)

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

        if new_board == self.board:
            # Nothing has happened
            return None
        else:
            return new_board

    def calculate_board_score(self, external_board=None):
        board = None
        if external_board:
            board = external_board
        else:
            board = self.board
        res = 0
        for l in board:
            r = copy.deepcopy(l)
            while ' ' in r:
                r.remove(' ')

            res += sum(r)
        return res

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

    def pprint(self):
        for line in self.board:
            print line

    def __repr__(self):
        if not self.score:
            return 'State score: {score}'.format(score=self.calculate_score())
        else:
            return 'State score: {score}'.format(score=self.score)