#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np


class Process:
    @staticmethod
    def in_expand(in_grid):
        shape = np.array(in_grid).shape
        if len(shape) > 1:
            shape_x, shape_y = shape

            if shape_x != shape_y:
                # This is already flattened
                return np.reshape(in_grid, (4, 4))
            else:
                return in_grid
        else:
            # This is already flattened
            return np.reshape(in_grid, (4, 4))

    @staticmethod
    def out_flatten(in_grid):
        return np.array(in_grid).flatten()

    @staticmethod
    def mergable_neighbours(in_grid):
        expanded = Process.in_expand(in_grid)
        result = np.zeros((4, 4))
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Loop through the grid
        for y in range(0, len(expanded)):
            for x in range(0, len(expanded[y])):
                # Set current value to 0
                current = expanded[y][x]

                # Loop throug changes in coordinates
                for direction in directions:
                    dy, dx = direction

                    # Check that values are in bounds
                    if 0 <= y + dy < len(expanded) and 0 <= x + dx < len(expanded[y]):
                        neighbour = expanded[y + dy][x + dx]
                        # If neighbour has same value, add one

                        if neighbour == current and current != float(0):
                            result[y][x] += 1

        return Process.out_flatten(result)

    @staticmethod
    def logarithm(in_grid):
        expanded = Process.in_expand(in_grid)
        result = np.zeros((4, 4))

        # Store highest
        highest = 0
        for y in range(len(expanded)):
            for x in range(len(expanded[y])):
                if expanded[y][x] > 0 and math.log(expanded[y][x], 2) > highest:
                    highest = math.log(expanded[y][x], 2)

        # Convert values
        for z in range(len(expanded)):
            for u in range(len(expanded[y])):
                if expanded[z][u] > 0:
                    result[z][u] = math.log(expanded[z][u], 2) / highest

        # Flatten and return
        return Process.out_flatten(result)

    @staticmethod
    def moveable_lines(in_grid):
        neighbours = Process.in_expand(Process.mergable_neighbours(in_grid))

        lines = np.zeros(8)

        # X-axis
        for y in range(len(neighbours)):
            addable_y = 0
            for x in range(len(neighbours[y])):
                if neighbours[y][x] > 0:
                    addable_y += 1
            if addable_y > 1:
                lines[y] = 1

        # Y-axis
        for _x in range(len(neighbours[0])):
            addable_x = 0
            for _y in range(len(neighbours)):
                if neighbours[_y][_x] > 0:
                    addable_x += 1
            if addable_x > 1:
                lines[_x * 2] = 1

        return Process.out_flatten(lines)

    @staticmethod
    def no_pro(in_grid):
        expanded = Process.in_expand(in_grid)
        return Process.out_flatten(expanded)

    @staticmethod
    def multiple_methods(argument=None, methods=None):
        result = None
        for method in methods:
            if result is None:
                result = method(argument)
            else:
                result = np.append(result, method(argument))


        return result.flatten()
