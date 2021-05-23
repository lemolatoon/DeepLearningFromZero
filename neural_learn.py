from numpy.core.numeric import cross
from functions import *
import numpy as np
import os, sys
from layer import Layer
sys.path.append(os.pardir)
from dataset.mnist import load_mnist



class Network:
    def __init__(self, layers):
        self.layers = layers
        for i, layer in enumerate(layers):
            Layer : layer
            #print(i)
            #print(i != len(layers))
            if i+1 != len(layers) and layer.output_num != layers[i + 1].input_num:
                print(str(i) + "番目は出力が" + str(layer.output_num) + "つなのに、" + str(i+1) + "番目の入力が" + "つで噛み合わないよ!")

    def predict(self, X):
        Y = X
        for layer in self.layers:
            Y = layer.calc(Y)
            #print(Y)
        return Y

    def get_parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[2*i - 1] = layer.ws #奇数番目が重み
            params[2*i] = layer.bs #偶数番目がバイアス
        return len(self.layers), params

    def set_parameters(self, params):
        for i, layer in enumerate(self.layers):
            layer.ws = params[2*i - 1] #奇数番目が重み
            layer.bs = params[2*i] #偶数番目がバイアス

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(x, t)

    def loss_by_param(self, x, t, params):
        tmp = self.get_parameters()
        self.set_parameters(params)
        y = self.predict(x)
        self.set_parameters(tmp)
        


    def backward(self, x, t):
        loss_W = lambda params: self.loss_by_param(x, t, params)
        grads = {}

        for i in len(self.layers):
            grads[2*i - 1] = numeral_gradient(loss_W, t)



def numeral_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size): #xの配列それぞれに対して勾配求めるよ
        tmp_val = x[idx] # 注意：xは参照渡しされている
        x[idx] = x + h
        fxh1 = f(x) # xは普通重みW   hだけ足されたx+hで計算されていることに注意

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 + fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

def creante_Network():
    return



if __name__ == "__main__":
    print("Start")
