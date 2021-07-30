import numpy as np
from util import im2col
from layers import Convolution
import os
from contextlib import redirect_stdout

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

dout = np.array([[2, 1],
                 [1, 1]])

image = image.reshape([1, 3, 4, 4])
filt = filt.reshape([1, 3, 3, 3])
dout = dout.reshape([1, 1, 2, 2])
dout = np.ones([1, 1, 2, 2])

b = 2

def convolution():
    conv = Convolution(filt.copy(), b)
    print(filt.shape)
    conv.forward(image)
    #print("==y==")
    #print(y)
    #print(y.shape)
    #print("==dout==")
    #print(dout) 
    conv.backward(dout)
    #print(conv)
    #print("=========Convolution===========")
    return conv

def compare():
    with redirect_stdout(open(os.devnull, "w")):
        conv1 = convolution()
        filt[0][0][0][0] = 1
        filt[0][0][0][1] = 5
        conv2 = convolution()
    print()
    print(conv1.W)
    print(conv1.dW)
    print(conv1.out)
    print()
    print(conv2.W)
    print(conv2.out)
    print()
    print(np.sum(np.abs(conv1.out - conv2.out)))

def main():
    with redirect_stdout(open(os.devnull, "w")):
        conv1 = convolution()
    conv1.print()
    





if __name__ == "__main__":
    main()
