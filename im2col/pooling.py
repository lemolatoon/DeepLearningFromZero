import numpy as np
from util import im2col
from layers import Pooling, Convolution

image = np.array([[[0, 0, 2, 0],
                   [2, 2, 3, 3],
                   [1, 2, 2, 3],
                   [0, 3, 1, 0]],
                  [[1, 2, 2, 2],
                   [2, 3, 3, 3],
                   [3, 0, 1, 2],
                   [2, 3, 0, 2]],
                  [[3, 3, 3, 1],
                   [3, 1, 0, 0],
                   [3, 0, 1, 2],
                   [0, 3, 0, 1]]])

filt = np.array([[[0, 4, 3],
                  [4, 2, 3],
                  [3, 2, 4]],
                 [[1, 2, 1],
                  [2, 1, 4],
                  [4, 1, 0]],
                 [[4, 2, 1],
                  [2, 3, 1],
                  [0, 3, 4]]])

dout = np.array([[1, 1],
                 [1, 1]])

image = image.reshape([1, 3, 4, 4])
filt = filt.reshape([1, 3, 3, 3])
dout = np.ones([1, 3, 2, 2])

b = 2


def pooling():
    pool = Pooling(2, 2)
    pool.forward(image)
    pool.backward(dout)
    return pool

def main():
    pool = pooling()
    pool.print()


if __name__ == "__main__":
    main()
