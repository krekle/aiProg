#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkinter import *
from tkFileDialog import askopenfilename
from board import Board
from run import Run
from functools import partial

"""
Gui class

The class creating a visual gui for the algorithm solving the problem
"""


class Gui(Tk):
    def __init__(self, *args, **kwargs):

        Tk.__init__(self, *args, **kwargs)

        # Force fullscreen
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth() - 3, self.winfo_screenheight() - 3))
        self.columnconfigure(0, weight=5)
        self.columnconfigure(1, weight=5)
        self.columnconfigure(2, weight=3)
        self.rowconfigure(0, weight=2)
        self.rowconfigure(1, weight=2)
        self.rowconfigure(2, weight=2)

        # Set the title
        self.title("AI-Prog Module 1 - Kristian Ekle & Thor Håkon")

        # Reference to menu
        self.menubar = Menu(self)

        # Reference to run class
        self.run = Run(self)

        # The sizof the squares in the map
        self.sqsize = 15

        # Defining the colors for the icons
        self.icon = {
            ".": "#CCFFFF",
            "#": "#4c4c4c",
            "S": "#FF6600",
            "G": "#33CC33",
            " ": "#CCFFFF",
            ".S": "#B8E6E6",
            ".G": "#99EB99"
        }



        # How long we should wait between each redraw
        self.delay = 20

        # Radiobuttons
        self.v1 = None
        self.v2 = None
        self.v3 = None
        self.v4 = None

        # Config
        self.algoritm = 0
        self.astar_stat = StringVar()
        self.bfs_stat = StringVar()
        self.dfs_stat = StringVar()

        # Populate the menues
        self.populate_menu()

    # Populates the menu
    def populate_menu(self):
        # Dummy
        self.v1 = StringVar()
        self.v2 = IntVar()
        self.v2.set(1)
        self.v3 = IntVar()
        self.v3.set(1)
        self.v4 = IntVar()
        self.v4.set(1)


        # Create a pulldown menu for chosing what level to play
        self.filemenu = Menu(self.menubar, tearoff=0)
        str_levels = self.run.list_files()

        for level in range(len(str_levels)):
            self.filemenu.add_radiobutton(label=str_levels[level][:-4],
                                          variable=self.v1,
                                          command=partial(self.play_level, str_levels[level]))
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Add New", command=lambda: self.open_file())
        self.menubar.add_cascade(label="Boards", menu=self.filemenu)

        # Create a pulldown menu for chosing delays
        self.delaymenu = Menu(self.menubar, tearoff=0)
        for i in ['0', '20', '50', '100', '200', '500', '1000']:
            self.delaymenu.add_radiobutton(label=i + " ms",
                                           variable=self.v4,
                                           value=i,
                                           command=lambda: self.set_delay(i))
        self.menubar.add_cascade(label="Delay", menu=self.delaymenu)

        # Menu element for closing the application
        self.menubar.add_command(label="Quit!", command=self.quit)

        # Apply the config
        self.config(menu=self.menubar)

    ## -----------------------------------------
    ##                Menu Handlers
    ## -----------------------------------------

    def play_level(self, level, str_level=None):


        # Set new run instance

        # Astar
        # Label for Algorithm
        astar_label = Label(master=self, text="Best First - Algorithm").grid(column=0, row=0, sticky=N+W, padx=20, pady=20)
        astar_generated = Label(master=self, textvariable=self.astar_stat).grid(column=0, row=2, sticky=N+W, padx=20, pady=20)

        astar_run = Run(self)
        self.astar_gui = Board(parent=self, stats_gui=self.astar_stat, run=astar_run, column=0, row=1, sq_size=self.sqsize, delay=self.delay, height=300, width=300)
        astar_run.gui = self.astar_gui


        # BFS
        # Label for Algorithm
        bfs_label = Label(master=self, text="BFS - Algorithm").grid(column=1, row=0, sticky=N+W, padx=20, pady=20)
        bfs_generated = Label(master=self, textvariable=self.bfs_stat).grid(column=1, row=2, sticky=N+W, padx=20, pady=20)

        bfs_run = Run(self)
        self.bfs_gui = Board(parent=self, stats_gui=self.bfs_stat, run=bfs_run, column=1, row=1, sq_size=self.sqsize, delay=self.delay, height=300, width=300)
        bfs_run.gui = self.bfs_gui

        # DFS
        # Label for Algorithm
        astar_label = Label(master=self, text="DFS - Algorithm").grid(column=2, row=0, sticky=N+W, padx=20, pady=20)
        dfs_generated = Label(master=self, textvariable=self.dfs_stat).grid(column=2, row=2, sticky=N+W, padx=20, pady=20)


        lifo_run = Run(self)
        self.lifo_gui = Board(parent=self, stats_gui=self.dfs_stat, run=lifo_run, column=2, row=1, sq_size=self.sqsize, delay=self.delay, height=300, width=300)
        lifo_run.gui = self.lifo_gui




        # Set the algorithm 0 = Astar, 1=BFS
        astar_run.set_algorithm(0)
        bfs_run.set_algorithm(1)
        lifo_run.set_algorithm(2)

        # Set the correct level in Run class
        if str_level:
            bfs_run.initialize(bfs_run.generate_board(str_level=str_level))
            astar_run.initialize((astar_run.generate_board(str_level=str_level)))
            lifo_run.initialize(lifo_run.generate_board(str_level=str_level))
        else:
            bfs_run.open_file(level)
            astar_run.open_file(level)
            lifo_run.open_file(level)

        # Run the solvers
        astar_run.run()
        bfs_run.run()
        lifo_run.run()

    def choose_algorithm(self, algorithm):
        self.algoritm = algorithm

    def set_delay(self, delay):
        self.delay = int(delay)

    def choose_module(self, module):
        print module

    def open_file(self):
        filename = askopenfilename(parent=self)
        f = open(filename)
        level = f.read()

        # start with chosen file
        self.play_level(None, level)

"""
Main method

Executing the entire application
"""


def main():
    # New instance of Gui
    app = Gui()

    # Start the mainloop
    app.mainloop()


"""
Executing Application
"""

main()
