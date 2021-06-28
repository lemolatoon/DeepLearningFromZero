import sys, os
import pandas as pd

from optimizers import *
sys.path.append(os.pardir)
from common.util import smooth_curve
import numpy as np
from dataset.mnist import load_mnist
from network import network
from layers import *
import matplotlib.pyplot as plt



def draw_loss_list(train_loss_list):
    x = np.arange(len(train_loss_list))
    plt.plot(x, train_loss_list, label='loss')
    plt.xlabel('iteration')
    plt.ylabel('acc')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

def draw():
    #path = "data/20210627_first/train_loss.pickle"
    path = "data/20210627/train_loss.pickle"
    train_loss = pd.read_pickle(path)
    draw_loss_list(train_loss)

def draw_test_train():
    #way = "Dropout"
    way = ""
    path1 = "data/20210627_{}/{}_{}.pickle".format(way, "train", "acc")
    path2 = "data/20210627_{}/{}_{}.pickle".format(way, "test", "acc")
    path1 = "data/20210627{}/{}_{}.pickle".format(way, "train", "acc")
    path2 = "data/20210627{}/{}_{}.pickle".format(way, "test", "acc")
    train = pd.read_pickle(path1)
    test = pd.read_pickle(path2)
    x = np.arange(len(train))
    plt.plot(x, smooth_curve(test), label="test")
    plt.plot(x, smooth_curve(train), label="train")
    plt.xlabel("iteration")
    plt.ylabel("acc")
    #plt.ylim(0, 1) #y軸の範囲
    #plt.xlim(0, 2000)
    plt.legend() #凡例
    plt.show()


def draw_lists():
    which = "acc"
    #which = "acc"
    train = "test"
    path1 = "data/20210627_Momentum/{}_{}.pickle".format(train, which)
    path2 = "data/20210627_SGD/{}_{}.pickle".format(train, which)
    path3 = "data/20210627_AdaGrad/{}_{}.pickle".format(train, which)
    Momentum = pd.read_pickle(path1)
    SGD = pd.read_pickle(path2)
    AdaGrad = pd.read_pickle(path3)
    x = np.arange(len(Momentum))
    plt.plot(x, smooth_curve(Momentum), label="Momentum")
    plt.plot(x, smooth_curve(SGD), label="SGD")
    plt.plot(x, smooth_curve(AdaGrad), label="AdaGrad")
    plt.xlabel("iteration")
    plt.ylabel(which)
    #plt.ylim(0, 1) #y軸の範囲
    #plt.xlim(0, 2000)
    plt.legend() #凡例
    plt.show()


if __name__ == "__main__":
    #draw_lists()
    draw_test_train()