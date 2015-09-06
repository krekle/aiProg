#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Tkconstants import BOTH, LEFT, W, E, S, N

from Tkinter import Canvas
import Tkinter
from module_1.state import NodeState


class Board(Canvas):
    icon = {
        ".": "#CCFFFF",
        "#": "#4c4c4c",
        "S": "#FF6600",
        "G": "#33CC33",
        " ": "#CCFFFF",
        ".S": "#B8E6E6",
        ".G": "#99EB99"
    }

    def __init__(self, stats_gui=None, parent=None, run=None, column=None, row=None, sq_size=15, delay=20, *args,
                 **kwargs):
        Canvas.__init__(self, parent, bg='white', highlightthickness=0, borderwidth=0, *args, **kwargs)
        self.sqsize = sq_size
        self.run = run
        self.delay = delay
        self.stats = stats_gui

        self.column = column
        self.row = row

    def draw(self, grid, finished, current, closed_list=None, open_list=None):
        # Kill if no grid was returned and finished flag is True
        if len(grid) is 0 and finished:
            # Draw the path
            return

        self.delete(Tkinter.ALL)

        # Use node class instead

        # Loop the grid
        for y in range(len(grid)):
            for x in range(len(grid[y])):
                # Define some sizes
                top = y * self.sqsize
                left = x * self.sqsize
                bottom = y * self.sqsize + self.sqsize - 2
                right = x * self.sqsize + self.sqsize - 2

                # Store current Node
                current_node = grid[y][x]

                # Test
                state = NodeState([y, x], None, None, current.goal)

                # Check if current Node is closed or not
                if not finished and state in open_list:
                    self.create_rectangle(left, top, right, bottom,
                                          outline="#ff0000",
                                          fill="green")
                    state = None
                elif current_node is '.':
                    self.create_rectangle(left, top, right, bottom,
                                          outline="#ff0000",
                                          fill=self.icon.get(current_node))
                else:
                    self.create_rectangle(left, top, right, bottom,
                                          outline=self.icon.get(current_node),
                                          fill=self.icon.get(current_node))

                # Check if on path
                # When finished, draw the path
                if finished:
                    if [y, x] in current.get_reconstructed_path():
                        self.create_oval(left, top, right, bottom, fill="#000")

        # Pack
        self.grid(row=self.row, column=self.column, sticky=N + S + E + W, padx=20, pady=20)

        # Wait and redraw if not finished
        if finished is False:
            self.after(self.delay, lambda: self.run.run())
        else:
            self.stats.set(
                'Length of path: ' + str(len(current.path)) + '\n' + 'Total nodes created: ' + str(len(closed_list)))
            print current.get_reconstructed_path()
