import sys, os

import pandas as pd

from optimizers import *
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from network import network
from layers import *
import matplotlib.pyplot as plt

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    net = network()

    net.add(Affine(None, None, input=784, output=50))
    #net.add(Relu())
    net.add(Dropout())
    net.add(Affine(None, None, input=50, output=10))
    net.lastLayer = SoftmaxWithLoss()

    net.weight_init()

    x = np.concatenate([x_train, x_test])
    t = np.concatenate([t_train, t_test])

    p = np.random.permutation(len(t_test)) #シャッフルしたindex
    x = x[p]
    t = t[p]

    print("x_shape: {}, t_shape: {}".format(x.shape, t.shape))

    net.learn(x, t, iters_num=100000, batch_size=1000, optimizer=SGD(lr=0.01))


if __name__ == "__main__":
    main()
