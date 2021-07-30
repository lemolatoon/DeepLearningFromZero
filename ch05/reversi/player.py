import numpy as np

class player:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy

    def next(self, board):
        """
        Input : board[8][8]

        Return : int x, int y
        """
        return self.strategy.next(board)

class strategy:
    def next(self, board):
        pass