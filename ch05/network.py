import os
import sys
import numpy as np
from collections import OrderedDict
from layers import *
from optimizers import *

import pandas as pd
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), "JST")


class network:
    """
    layers : OrderedDict
    lastLayer : softmaxWithLayer とか
    """
    def __init__(self):
        self.layers = OrderedDict()
        self.lastLayer = SoftmaxWithLoss
        self.params = {}


    def weight_init(self, weight_init_std=0.01):
        #weight initialization

        for key in self.layers.keys():
            layer = self.layers[key]
            num = 0
            if type(layer) is Affine:
                num += 1
                layer.weight_init(weight_init_std)
                self.params["W" + str(num) + " of " + key] = layer.W
                self.params["b" + str(num) + " of " + key] = layer.b

        print(self.layers)

    def add(self, layer):
        index = len(self.layers)
        self.layers[str(layer) + str(index + 1)] = layer

    def predict(self, x, train_flg=True):
        #print("forwarding...")
        for layer in self.layers.values():
            #print("x.shape: {}".format(x.shape))
            if type(layer) is Dropout : layer.train_flg = train_flg
            x = layer.forward(x)
        #print("forward ended")

        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg=train_flg)
        #print("lastLayer y.shape: {}".format(y.shape))
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        #forward
        self.loss(x, t)

        #backward
        #print("backwarding...")
        dout = 1
        dout = self.lastLayer.backward(dout)
        #print("lastLayer dout.shape: {}".format(dout.shape))

        layers = list(self.layers.values())
        #print(layers)
        layers.reverse()
        #print(layers)
        for layer in layers:
            #print("input: {}, output: {}".format(layer.input, layer.output))
            dout = layer.backward(dout)

        grads = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            num = 0
            if type(layer) is Affine:
                num += 1
                grads["W" + str(num) + " of " + key] = layer.dW
                grads["b" + str(num) + " of " + key] = layer.db

        return grads

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy


    def learn(self, x, t ,iters_num=10000, batch_size=100, optimizer=SGD()):
        rate = 0.7
        x_train = x[:int(len(x) * rate)]
        x_test = x[int(len(x) * rate):]
        t_train = t[:int(len(t) * rate)]
        t_test = t[int(len(t) * rate):]

        train_size = x_train.shape[0]

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        print(optimizer.lr)

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            #勾配求める
            #print("x_batch: {}, t_batch: {}".format(x_batch.shape, t_batch.shape))
            grads = self.gradient(x_batch, t_batch)
            optimizer.update(self.params, grads)

            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                print("index: {}".format(i))
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("train_acc : {} test_acc : {}".format(train_acc, test_acc))

        dt_now = datetime.now(JST)
        path = dt_now.strftime("%Y%m%d")
        path = "./data/{}".format(path)
        #path = "a/1"
        print(path)
        #os.system("mkdir -p {}".format(path))
        os.makedirs(path, exist_ok=True)
        pd.to_pickle(self.params, path + "/params.pickle")
        pd.to_pickle(train_loss_list, path + "/train_loss.pickle")
        pd.to_pickle(train_acc_list, path + "/train_acc.pickle")
        pd.to_pickle(test_acc_list, path + "/test_acc.pickle")




if __name__  == "__main__":
    a = Affine(1, 2, 3, 5)
    #print(type(a))
    """
    print(str(type(a)))
    print(str(a))
    print(str(type(a)))
    print(type(Affine))
    print(type(a) is Affine)
    print(type(a) is type(Affine))
    """
    """
    a = {}
    a["1"] = 1
    a["2"] = 2
    print(len(a))
    """

    a = np.array([[1, 2, 3], [4, 5, 6]])
    x_train = a[:int(len(a) * 0.3)]
    print(len(a))
    print(int(len(a) * 0.3))
    print(x_train)

    

