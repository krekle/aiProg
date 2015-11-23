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
