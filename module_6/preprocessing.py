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
    def mergable_neighbours(grid, nested=False, shape=(4, 4)):
        """
        Counts number of similar neighbours for a tile
        :param grid:
        :param nested:
        :return:
        """
        # If nested then pass through multi_runner
        if nested:
            return Process.multi_runner(grid, Process.mergable_neighbours, shape=shape)
        else:
            grid = np.array(grid).reshape(shape)

            result = np.zeros(shape=(len(grid), len(grid[0])))

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            # Loop through the grid
            for y in range(0, len(grid)):
                for x in range(0, len(grid[y])):
                    # Set current value to 0
                    current = grid[y][x]

                    # Loop throug changes in coordinates
                    for direction in directions:
                        dy, dx = direction

                        # Check that values are in bounds
                        if 0 <= y+dy < len(grid) and 0 <= x+dx < len(grid[y]):
                            neighbour = grid[y + dy][x + dx]
                            # If neighbour has same value, add one

                            if neighbour == current and current is not 0:
                                result[y][x] += 1

            return result

#test = np.array([[2, 1, 0, 2, 2, 1, 2, 2, 1]])


#results = Process.mergable_neighbours(test, shape=(3, 3))
#print(results)
