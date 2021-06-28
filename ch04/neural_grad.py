import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
from functions import *
import pandas as pd
from common.functions import sigmoid_grad 
import matplotlib.pyplot as plt



def numerical_diff(f, x):
    '''
    h = 1e-7
    return (f(x+h) - f(x-h)) / (2*h)
    '''

    h = 1e-4
    grad = np.zeros_like(x)
    print("微分中")
    print("x.shape: " + str(x.shape))

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"]) #(0, 0), (0, 1), (1, 0), (1, 1)のようにxの配列のインデックスを順に返す（多次元でも！）

    while not it.finished:
        i = it.multi_index
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp

        it.iternext()

    print("微分終了")
    return grad

def image_show(img):
    print(img.shape)
    if img.shape[0] > 28:
        img = img.reshape(28, 28)
        print("reshapeしたよ")
        print(img.shape)
    if np.sum(img) <= img.size:
        img = img * 255.0
        print("正規化戻したよ")
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()

class Network():
    def __init__(self, input=784, hidden=100, output=10):
        self.grad = {}

        self.param = {}
        self.param["W1"] = np.random.randn(input, hidden)
        self.param["b1"] = np.zeros(hidden)
        self.param["W2"] = np.random.randn(hidden, output)
        self.param["b2"] = np.zeros(output)
    
    def predict(self, x):
        W1 = self.param["W1"]
        b1 = self.param["b1"]
        W2 = self.param["W2"]
        b2 = self.param["b2"]

        x = np.dot(x, W1) + b1
        x = sigmoid(x)

        x = np.dot(x, W2) + b2
        x = softmax(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t) #呼び出されたときに実行される

        grad = {}

        grad["W1"] = numerical_diff(loss_W, self.param["W1"]) #self.param参照渡し paramを少しずらしたときの勾配（偏微分）
        grad["b1"] = numerical_diff(loss_W, self.param["b1"])
        grad["W2"] = numerical_diff(loss_W, self.param["W2"])
        grad["b2"] = numerical_diff(loss_W, self.param["b2"])

        return grad


    def gradient(self, x, t):
        W1, W2 = self.param['W1'], self.param['W2']
        b1, b2 = self.param['b1'], self.param['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        #print("accuracy x: " + str(x))
        #print("accuracy t: " + str(t))
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0]) #分母は実質ミニバッチ
        return accuracy

    def train(self, x, t, x_test, t_test, batch_size=100):
        train_size = x.shape[0]
        iter_num = 100000
        learning_rate = 0.1
        iter_per_epoch = max(train_size / batch_size, 1)

        train_accs = []
        test_accs = []
        train_losses = []
        test_losses = []
        train_data = {}

        for i in range(iter_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x[batch_mask]
            t_batch = t[batch_mask]

            #grad = self.numerical_gradient(x_batch, t_batch)
            grad = self.gradient(x_batch, t_batch)

            for key in ("W1", "b1", "W2", "b2"):
                self.param[key] -= learning_rate * grad[key]

            if i % iter_per_epoch == 0:
                pd.to_pickle(self.param, "param_tmp.pkl")

                train_acc = self.accuracy(x, t)
                test_acc = self.accuracy(x_test, t_test)
                train_loss = self.loss(x, t)
                test_loss = self.loss(x_test, t_test)

                train_accs.append(train_acc)
                test_accs.append(test_acc)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                
                print(str(i) + "th   train_acc: " + str(train_acc) + "  test_acc: " + str(test_acc))
            
            train_data = {
                "train_acc" : train_accs,
                "test_acc" : test_accs,
                "train_loss" : train_losses,
                "test_losses" : test_losses
            }

        self.show_data(train_data)
        pd.to_pickle(train_data, "train_data.pkl")
        pd.to_pickle(self.param, "param.pkl")


    def load(self, param=None):
        if param is None:
            self.param = pd.read_pickle("param.pkl")
        else: 
            self.param = pd.read_pickle(param)

    def show_data(train_data=None):
        if train_data is None:
            train_data = pd.read_pickle("train_data.pkl")
        train_acc = train_data["train_acc"]
        test_acc = train_data["test_acc"]
        train_loss = train_data["test_acc"]
        test_loss = train_data["test_loss"]

        i = range(0, 10000, 100)
        plt.plot(i, train_acc, label="train acc")
        plt.plot(i, test_acc, label="test acc", linestyle="--")
        plt.ylim(0, 1.0)
        plt.show()

    
        plt.plot(i, train_loss, label="train loss")
        plt.plot(i, test_loss, label="test loss", linestyle="--")
        #plt.ylim(0, 1.0)
        plt.show()



def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    net = Network(input=784, hidden=100, output=10)
    net.load("param_tmp.pkl")
    #net.load()
    i = np.random.choice(t_test.shape[0]) 
    y = net.predict(x_test[i])
    y = np.argmax(y)
    print("y: " + str(y))
    print("t: " + str(t_test[i]))

    print(net.accuracy(x_test, t_test))

    #grad = net.numerical_gradient(x_test[0:i], t_test[0:i])
    #net.train(x_train, t_train, x_test, t_test)

    print(net.accuracy(x_test, t_test))

if __name__ == "__main__":
    main()