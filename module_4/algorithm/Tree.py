from module_4.game.board import Direction


class TreeNode():

    """
    Constructor chain creates children until deep is 0
    """

    def __init__(self, this, parent, deep, choice=None, mx=True):
        self.parent = parent    # The parent TreeNode of this node, none = this is the root
        self.children = []      # The children TreeNodes of this Treenode
        self.this = this        # The actual board-state of this TreeNode
        self.mx = mx            # Boolean if this is a max or min level
        self.choice = choice    # Choice, if this is a choice-level node, stores Up, Down etc

        if deep == 0:           # Count how deep the tree goes
            # If this is a leaf node
            self.score = this.calculate_score()

        else:
            # If parent = None -> this is start State
            if mx:
                kids = this.max_successors()
                for key in kids.keys():
                    self.children.append(TreeNode(kids[key], self, deep - 1, choice=key, mx=not self.mx))
            else:

                for child in this.min_successors():
                    n = TreeNode(child, self, deep - 1, not self.mx)
                    self.children.append(n)


            # Calculate the node score after adding all children
            self.score = self.node_score()
            # First max -> min

    def get_min(self):
        # return min of children
        min = None
        for child in self.children:
            if min is None:
                min = child
            if child.score < min.score:
                min = child
        return min

    def get_avg(self):
        # return min of children
        sum = 0
        for child in self.children:
            sum += child.score
        return sum / len(self.children)

    def get_max(self):
        # return max of children
        max = None
        for child in self.children:
            if max is None:
                max = child
            if child.score > max.score:
                max = child
        return max

    def get_move(self):
        movement_state = None
        if self.mx:
            movement_state = self.get_max()
        else:
            movement_state = self.get_min()

        # No possible moves, game over
        if movement_state is None:
            return None, None

        return Direction.get(movement_state.choice)

    def node_score(self):
        # Check if this is max or min dept
        if len(self.children) > 0:
            if self.mx:
                return self.get_max().score
            else:
                # return self.get_min().score
                # Changed to average, better results
                return self.get_avg()
        else:
            return self.this.calculate_score()

    def __repr__(self):
        return 'NodeScore: {score}, Current: {this}'.format(score=str(self.score),
                                                            this=str(self.this))
