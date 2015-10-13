#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from module_4.game.board import Direction, Game

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
        score += 20 * free
        score += highest_tile * 5

        # check number of adjecent that can be combined next time
        # High tile with only 2 neighbours = corner

        # Snake: Top left is worth the most
        snake = [60, 35, 30, 25]
        for i in range(len(snake)):
            ch = self.board[0][i]
            if str(ch).isdigit():
                score += int(ch) * snake[i]

        snake2 = [2, 4, 7, 12]
        for k in range(len(snake2)):
            ch = self.board[1][k]
            if str(ch).isdigit():
                score += int(ch) * snake2[k]


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


class TreeNode():

    def __init__(self, this, parent, deep, choice=None, mx=True):
        self.parent = parent
        self.children = []
        self.this = this
        self.mx = mx
        self.choice = choice

        if deep == 0:
            # If this is a leaf node
            self.score = this.calculate_score()

        else:
            # If parent = None -> this is start State
            if mx:
                kids = this.max_successors()
                for key in kids.keys():
                    self.children.append(TreeNode(kids[key], self, deep - 1, choice=key, mx=not self.mx))
            else:

                for child in this.min_successors():
                    n = TreeNode(child, self, deep - 1, not self.mx)
                    self.children.append(n)


            # Calculate the node score after adding all children
            self.score = self.node_score()
            # First max -> min

    def get_min(self):
        # return min of children
        min = None
        for child in self.children:
            if min is None:
                min = child
            if child.score < min.score:
                min = child
        return min

    def get_avg(self):
        # return min of children
        sum = 0
        for child in self.children:
            sum += child.score
        return sum/len(self.children)

    def get_max(self):
        # return max of children
        max = None
        for child in self.children:
            if max is None:
                max = child
            if child.score > max.score:
                max = child
        return max

    def get_move(self):
        movement_state = None
        if self.mx:
            movement_state = self.get_max()
        else:
            movement_state =  self.get_min()

        return Direction.get(movement_state.choice)

    def node_score(self):
        # Check if this is max or min dept

        if len(self.children) > 0:
            if self.mx:
                return self.get_max().score
            else:
                #return self.get_min().score
                return self.get_avg()
        else:
            return self.this.calculate_score()

    def __repr__(self):
        return 'NodeScore: {score}, Current: {this}'.format(score=str(self.score),
                                                                            this=str(self.this))
