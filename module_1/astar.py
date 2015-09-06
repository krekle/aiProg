__author__ = 'krekle'

from heapq import heappush, heappop, heapify


class Astar():
    def __init__(self, sorting=0, start_state=None):

        self.closed_list = set()    # List for explored states
        self.sorting = sorting      # Store the sorting method
        if sorting is 0:            # If we're using best-first (ASTAR)
            self.open_list = []
            heapify(self.open_list)
        else:                       # # DFS or LIFO
            self.open_list = []

        if start_state is not None: # Add start node
            self.to_explore(start_state)

        self.finished = False       # Status
        self.generated = {}         # Already generated nodes

    # Items to be explored are stored in the open_list
    def to_explore(self, node):
        if self.sorting is 0:
            heappush(self.open_list, node)
        else:
            self.open_list.append(node)

    # Explored are stored in the closed_list
    def explored(self, node):
        self.closed_list.add(node)

    # Get the next item to be explored, based on the sorting algorithm
    def get_explore(self):
        if self.sorting == 0:  # Best-first
            return heappop(self.open_list)
        elif self.sorting == 1:  # BFS
            return self.open_list.pop(0)
        elif self.sorting == 2:  # DFS
            return self.open_list.pop(-1)

    def attach_and_eval(self, child, current):
        child.parent = current
        child.g = current.g + current.distance_to(child)
        child.calculate_h()

    def do_next(self, gui):

        # Check that the open_list contains eligible next states
        if len(self.open_list) is 0:
            print 'No path avaliable!'

        # Get the next state from the openlist
        current = self.get_explore()

        # Add the current node to closed list
        self.closed_list.add(current)
        current.close()

        # Check goal, if succeed -> finish
        if current.is_goal():
            print 'Goal state found!'
            gui.draw(current.board, current.is_goal(), current, self.closed_list)

        # Generate children
        for child in current.generate_successors():
            # check if state has been
            # if generated_nodes.get(child.hash):

            # If it is already generated, take from the generated dict
            if self.generated.get(child.__hash__()) is not None:
                child = self.generated.get(child.__hash__())

                self.cascade_generated(current, child)
            else:
                # Successor is not generated
                self.generated[child.__hash__()] = child
                self.attach_and_eval(child, current)
                self.to_explore(child)

            current.children.append(child)

        gui.draw(current.board, current.is_goal(), current, self.closed_list, self.open_list)
        return current

    def cascade_generated(self, current, child):
        g = current.g + current.distance_to(child)

        if g < child.g:
            self.attach_and_eval(child, current)
            if child in self.closed_list:
                self.propagate_path_improvement(child)
            # We have to resort our heap when we internally change a value the heap is sorted on.
            if self.sorting == 0:
                self.open_list.sort()

    # Update all g values and parents along a new improved path
    def propagate_path_improvement(self, state):
        for child in state.children:
            new_g = state.g + state.distance_to(child)
            if new_g < child.g:
                child.parent = state
                child.g = new_g
                self.propagate_path_improvement(child)
