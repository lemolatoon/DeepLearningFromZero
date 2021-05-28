# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sum_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t.reshape(1, t.size)
        y.reshape(1, y.size)

    batch_size = y.shape[0]
    #print("batch_size: " + str(batch_size))

    if t.size == y.size: #sizeは全要素数    つまりone-hotのとき
        t = np.argmax(t, axis=1)

    return -np.sum(np.log(y[np.arange(batch_size), t]))


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)