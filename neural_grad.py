import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image
from functions import *

def numerical_diff(f, x):
    '''
    h = 1e-7
    return (f(x+h) - f(x-h)) / (2*h)
    '''

    h = 1e-4
    grad = np.zeros_like(x)
    print("微分中")

    for i in range(x.shape[1]):
        print(i)
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp

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

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0]) #分母は実質ミニバッチ
        return accuracy



def main():
    (x_train, t_train), (x_test, t_test) = load_mnist()
    net = Network(input=784, hidden=100, output=10)
    i = 3 
    y = net.predict(x_test[i])
    y = np.argmax(y)
    print("y: " + str(y))
    print("t: " + str(t_test[i]))

    grad = net.numerical_gradient(x_test[0:i], t_test[0:i])

if __name__ == "__main__":
    main()