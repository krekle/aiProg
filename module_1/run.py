"""
Run class

The class opening files etc
"""
import glob
import os
from astar import Astar
from state import NodeState


class Run:
    # Constructor
    def __init__(self, gui):

        self.gui = gui

        # Algorithm to use, default = 0 -> astar
        self.algorithm_choice = 0
        self.algorithm = None

    # Set algorithm
    def set_algorithm(self, type):
        self.algorithm_choice = type
        if self.algorithm:
            self.algorithm.sorting = self.algorithm_choice

    def open_file(self, level):
        self.initialize(self.generate_board(level))

    def initialize(self, specs):
        # Create start end goal states
        start = NodeState(specs.get('start'), None, specs.get('board'), specs.get('goal'))

        # Set the algorithm, with start and goal
        self.algorithm = Astar(start_state=start, sorting=self.algorithm_choice)

    def __delete__(self, instance):
        print 'del run'
        del self.algorithm

    # Run algorithm
    def run(self):

        # Calculate scores
        self.algorithm.do_next(self.gui)


    ## ----------------------------------
    ##   File and Directory operations
    ## ----------------------------------


    # Read all avaliable boards
    def list_files(self):
        levels = []
        os.chdir("boards")
        for file in glob.glob("*.txt"):
            if file:
                levels.append(file)
        return levels

    # Read specific file
    def read_file(self, filename):
        with open(os.path.dirname(os.path.realpath(__file__)) + "/boards/" + filename, "r") as myfile:
            data = myfile.read()
            return data

    # Save a new level file
    def save_file(self, filename, string_file):
        text_file = open("boards/" + filename, "w")
        text_file.write(string_file)
        text_file.close()

    # Create a board from text
    def generate_board(self, filename=None, str_level=None):
        board = None

        if str_level is not None:
            raw_board = str_level
        else:
            raw_board = self.read_file(filename)

        board_config = raw_board.split('\n')

        # Initialize board
        board_size = board_config[0].split(' ')
        board = [[' ' for y in range(int(board_size[1]))] for x in range(int(board_size[0]))]

        # Start and Goal
        board_start = [int(board_config[1].split(' ')[0]), int(board_config[1].split(' ')[1])]
        board_goal = [int(board_config[1].split(' ')[2]), int(board_config[1].split(' ')[3])]
        board[board_start[0]][board_start[1]] = 'S'
        board[board_goal[0]][board_goal[1]] = 'G'

        # Constraints, 2 first lines are size and start, goal
        for i in range(2, len(board_config)):
            constraint_line = board_config[i].split(' ')
            constraint_origo = [int(constraint_line[0]), int(constraint_line[1])]
            constraint_dimen = [int(constraint_line[2]), int(constraint_line[3])]

            for cx in range(constraint_origo[0], constraint_dimen[0] + constraint_origo[0]):
                for cy in range(constraint_origo[1], constraint_dimen[1] + constraint_origo[1]):
                    board[cx][cy] = '#'

        board_specs = {'board': board, 'start': board_start, 'goal': board_goal}
        return board_specs
