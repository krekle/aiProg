#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Process:
    @staticmethod
    def multi_runner(data, function, **kwargs):
        """
        Function for calling methods with multiple arrays
        :param data:
        :param function:
        :return:
        """
        results = []
        for d in data:
            results.append(function(d, kwargs))
        return results

    @staticmethod
    def mergable_neighbours(in_grid, nested=False, shape=(4, 4)):
        """
        Counts number of similar neighbours for a tile
        :param grid:
        :param nested:
        :return:
        """
        # If nested then pass through multi_runner
        if nested:
            return Process.multi_runner(in_grid, Process.mergable_neighbours, shape=shape)
        else:
            copy = []
            if type(in_grid) == np.ndarray:
                copy = np.reshape(in_grid, (4, 4))
            else:
                copy = np.array(in_grid).reshape(shape)

            result = np.zeros(shape=(len(copy), len(copy[0])))

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            # Loop through the grid
            for y in range(0, len(copy)):
                for x in range(0, len(copy[y])):
                    # Set current value to 0
                    current = copy[y][x]

                    # Loop throug changes in coordinates
                    for direction in directions:
                        dy, dx = direction

                        # Check that values are in bounds
                        if 0 <= y+dy < len(copy) and 0 <= x+dx < len(copy[y]):
                            neighbour = copy[y + dy][x + dx]
                            # If neighbour has same value, add one

                            if neighbour == current and current != float(0):
                                result[y][x] += 1

            return result
