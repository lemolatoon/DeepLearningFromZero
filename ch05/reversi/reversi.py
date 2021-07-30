import numpy as np

class board:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.board = np.zeros([8, 8])
        self.board = np.random.randn(8, 8).astype("int")
        self.board[3][3] = 1 #1が黒, 2が白, 0が空白　横がx, 縦がy
        self.board[3][4] = 2
        self.board[4][3] = 2
        self.board[4][4] = 1

    def set(self, x, y, color):
        self.board[x][y] = color

    def settable(self, color):
        b = self.board.copy()
        try:
            #空白かどうかチェック
            room = self.board[x, y] == 0
            vertical = self.board

        except IndexError as e:
            #indexerrorのときはすべてfalse
            print(e)
            f = np.zeros_like(x, dtype="bool")
            return f

    def action(self, x, y, color):
        k = 8 - x
        column = self.board[x] #行
        row = self.board[y] #列
        left_upDiag = self.board.diagonal(offset=y-x) #offsetは対角成分の場所
        right_upDiag = (np.fliplr(self.board)).diagonal(offset=-7+x+y) #左右反転してから対角成分をとる
        return right_upDiag




class master:
    def __init__(self, player1, player2):
        
        self.pr1 = player1
        self.pr2 = player2
        self.board = None

if __name__ == "__main__":
    """
    b = board()
    a = np.arange(64)
    a = a.reshape((8, 8))
    """
    b = np.arange(64)
    b.reshape((8, 8))
    x = np.arange(8)
    y = np.arange(8)
    b = board()
    k = b.board[x][y] == 0
    b.board[k] = 56
    print(b.board)
    print(b.board[x][y] == 0)
    print("============================")
    print(b.board[np.array([[3], [4]])])
    print("============================")
    x, y = np.arange(8), np.arange(8)
    print(x)
    print(y)
    print("============================")
    b = board()
    print(b.board)
    print("============================")
    print(np.fliplr(b.board))
    x,y = 3, 4
    print("-7+x+y={}".format(-7+x+y))
    print(b.action(x, y, color=1))

