#!/usr/bin/env python
# -*- coding: utf-8 -*-

from Tkconstants import S, E, W, N

from Tkinter import Tk, Frame, Label, Button, Toplevel, Canvas, StringVar
from functools import partial
import random
from module_4.algorithm.algorithm import MinMax
from module_4.game.board import Game, Direction

__author__ = 'krekle'

sq_size = 130
colors = {
    ' ': '#FFFFFF',
    2: '#CCFFFF',
    4: '#66FF99',
    8: '#33CC33',
    16: '#33CCFF',
    32: '#33CCCC',
    64: '#CCCC00',
    128: '#006699',
    256: '#FF66CC',
    512: '#FF0066',
    1024: '#FFFF00',
    2048: '#9933FF',
    4096: '#FFFFCC',
    8192: '#003300'
}


class GameGui(Tk):
    score = None

    def __init__(self, parent):
        Tk.__init__(self)
        self.parent = parent

        # Game title
        self.title("2048")

        # Geometry
        self.geometry("{0}x{1}+0+0".format((self.winfo_screenwidth() / 2) - 300, (self.winfo_screenheight() / 2) + 200))
        self.columnconfigure(1, weight=5)

        # Game
        self.game = Game()

        # MinMax
        self.algorithm = None

        # Key Bindings
        self.bind('<Left>', self.move)
        self.bind('<Right>', self.move)
        self.bind('<Up>', self.move)
        self.bind('<Down>', self.move)
        self.bind('<n>', self.move)
        self.bind('<a>', self.move)

        #  Outline of the gui
        #########################
        ##  Game Score here    ##
        #########################
        ##  Game Board here    ##
        #########################
        ##  Menu Button here   ##
        #########################

        ## Score ##

        self.score_label = Label(self, text='Score', font=("Helvetica", 32, "bold")).grid(row=0, column=0, padx=20,
                                                                                          pady=20)
        self.score = Label(self, text="0", font=("Helvetica", 32, "bold")).grid(row=0, column=1)

        ## Board ##
        self.board = Canvas(self, width=(self.winfo_screenwidth() / 2) - 400, height=(self.winfo_screenheight() / 2),
                            highlightthickness=0, borderwidth=0)
        self.board.grid(row=1, column=0, columnspan=2, sticky=E + W, padx=10, pady=10)

        ## Menu ##
        btn = Button(self, text="Close", command=partial(parent.openMenuFrame, self))
        btn.grid(row=2, column=0)

        btn = Button(self, text="Close", command=partial(parent.openMenuFrame, self))
        btn.grid(row=2, column=1)

        # Draw the initial grid
        self.draw(self.game.grid)

    def draw(self, grid):

        for y in range(0, 4, 1):
            for x in range(0, 4, 1):
                top = y * sq_size
                left = x * sq_size
                bottom = y * sq_size + sq_size - 2
                right = x * sq_size + sq_size - 2

                # Create tiles and text
                item = self.board.create_rectangle(left, top, right, bottom)

                # Add the items
                self.board.itemconfig(item, fill=colors[grid[y][x]], outline="#ff0000")
                self.board.create_text((right - sq_size) + (sq_size / 2), (top + sq_size) - (sq_size / 2),
                                       text=str(grid[y][x]), font=("Helvetica", 32, "bold"))
                # self.board.itemconfig(item_text)
                # self.board.create_rectangle(left, top, right, bottom, fill=colors[grid[y][x]])

    def move(self, key, keycode=None):
        if keycode:
            code = keycode
        else:
            code = key.keycode

        if code == 8320768:
            self.game.move(Direction.Up)
        elif code == 8189699:
            self.game.move(Direction.Right)
        elif code == 8255233:
            self.game.move(Direction.Down)
        elif code == 8124162:
            self.game.move(Direction.Left)
        # Algorithm
        # one next
        elif code == 2949230:
            if not self.algorithm:
                self.algorithm = MinMax(self, self.game)
            dir = self.algorithm.run()
            self.game.move(dir)
        # auto
        elif code == 97:
            if not self.algorithm:
                self.algorithm = MinMax(self, self.game)
            self.auto()

        self.draw(self.game.grid)
        self.score = Label(self, text=str(self.game.calculate_score()), font=("Helvetica", 32, "bold")).grid(row=0,
                                                                                                             column=1)

    def auto(self):
        dir = self.algorithm.run()
        self.game.move(dir)
        self.draw(self.game.grid)
        self.after(200, lambda: self.auto())


class MainMenu(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)

        # Force fullscreen
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth() / 3, self.winfo_screenheight() / 2))

        # Set the title
        self.title("AI-Prog Module 1 - Kristian Ekle & Thor Håkon")


        # Bring window to front
        self.lift()
        self.attributes('-topmost', True)
        self.attributes('-topmost', False)

        btn = Button(self, text="Start Game", command=self.openGameFrame)
        btn.pack()

    def hide(self):
        self.withdraw()

    def openGameFrame(self):
        self.hide()
        subFrame = GameGui(self)

    def openMenuFrame(self, otherFrame):
        otherFrame.destroy()
        self.show()

    def show(self):
        self.update()
        self.deiconify()
