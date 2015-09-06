#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import fabs


class State(object):
    f = float()  # - estimated total cost of a solution path going through this node; f = g + h
    h = float()  # - estimated cost to goal
    g = float()  # - cost of getting to this node

    parent = None  # - pointer to best parent node
    children = []  # - list of all successor nodes, whether or not this node is currently their best parent
    value = None
    path = []
    goal = None
    hash = None

    state = None  # - an object describing a state of the search process
    closed = False

    def __init__(self, value, parent, goal, **kwargs):
        self.value = value

        if parent:
            self.path = parent.path[:]
            self.path.append(self.value)
            self.goal = parent.goal
        else:
            self.goal = goal
            self.path.append(self.value)

    def generate_successors(self):
        pass

    def calculate_f(self):
        self.f = self.g + self.h

    def calculate_g(self):
        self.g = len(self.path)

    def calculate_h(self):
        pass

    def is_goal(self):
        if self.value[0] == self.goal[0] and self.value[1] == self.goal[1]:
            return True
        return False

    def __repr__(self):
        return '({x}, {y})'.format(x=self.value[0], y=self.value[1])

    def __eq__(self, other):
        return self.value[0] == other.value[0] and self.value[1] == other.value[1]

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif self.f > other.f:
            return False
        else:
            if self.h >= other.h:
                return True
            else:
                return False

    def __hash__(self):
        return hash(self.__repr__())


class NodeState(State):
    board = [[]]

    # TWEAK:
    distance = 0

    def __init__(self, value, parent, board, goal):
        super(NodeState, self).__init__(value, parent, goal)
        if parent:
            self.board = parent.board
        else:
            self.board = board

        self.calculate_h()
        self.calculate_g()
        self.calculate_f()

    def calculate_h(self):
        self.h = float(fabs(self.value[0] - self.goal[0]) + fabs(self.value[1] - self.goal[1]))
        return self.h

    def manhattan_distance(self, other):
        return float(fabs(self.value[0] - other.value[0]) + fabs(self.value[1] - other.value[1]))

    def distance_to(self, other):
        if self.distance is 0:
            return self.manhattan_distance(other)

    def generate_successors(self):
        successors = []

        x = self.value[0]
        y = self.value[1]

        # Up, Down, Left, Right -> Possible successors
        successor_coords = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]

        # Check that the successors are in bounds and return
        for coords in successor_coords:
            if self.validate_successor(*coords):
                successors.append(self.validate_successor(*coords))

        return successors

    def validate_successor(self, x, y):
        # Check that the permutation is in bounds
        try:

            # Check that it is not a wall
            if self.board[x][y] is not '#':
                if x >= 0 and y >= 0:
                    return NodeState([x, y], self, None, None)
            else:
                # This is a wall, do nothing
                pass
        except:
            pass

    def close(self):
        if self.board[self.value[0]][self.value[1]] is 'S':
            self.board[self.value[0]][self.value[1]] = '.S'
        elif self.board[self.value[0]][self.value[1]] is 'G':
            self.board[self.value[0]][self.value[1]] = '.G'
        else:
            self.board[self.value[0]][self.value[1]] = '.'
        self.closed = True

    def get_reconstructed_path(self):
        path = [self]
        value_path = [self.value]
        while path[-1].parent:
            path.append(path[-1].parent)

        for state in path:
            value_path.append(state.value)
        return list(reversed(value_path))
