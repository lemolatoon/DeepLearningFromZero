import numpy as np
import perceptron as pcp

class Layer:
    def __init__(self, num, ws, bs, activation=None):
        self.num = num
        self.ws = ws
        self.bs = bs
        if activation is not None:
            self.activation = activation


    def calc(self, xs):
        as = np.dot(self.ws, xs) + self.bs
        return self.activation(as)


    def activation(self, as):
        return Layer.sigmoid(as)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))