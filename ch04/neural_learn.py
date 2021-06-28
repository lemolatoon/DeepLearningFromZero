from PIL import Image
from numpy.core.numeric import cross
from functions import *
import numpy as np
import os, sys
sys.path.append(os.pardir)
from dataset.mnist import load_mnist



class Network:
    def __init__(self, layers):
        self.params = {}
        self.layer_num = len(layers)

        
        self.layers = layers
        for i, layer in enumerate(layers):
            Layer : layer

            self.params["W" + str(i)] = np.random.randn(layer.input_num, layer.output_num)
            self.params["b" + str(i)] = np.zeros(layer.input_num)

            if i+1 != len(layers) and layer.output_num != layers[i + 1].input_num:
                print(str(i) + "番目は出力が" + str(layer.output_num) + "つなのに、" + str(i+1) + "番目の入力が" + "つで噛み合わないよ!")

    def predict(self, X, update_param=True):
        if update_param:
            self.set_parameters(self.params)
        Y = X
        for layer in self.layers:
            Y = layer.calc(Y)
            #print(Y)
        return Y

    def set_parameters(self, params):
        for i, layer in enumerate(self.layers):
            layer.ws = params["W" + str(i)] #奇数番目が重み
            layer.bs = params["b" + str(i)] #偶数番目がバイアス

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(x, t)

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}

        for i in range(0, self.layer_num):
            grads["W" + str(i)] = numerical_gradient(loss_W, self.params["W" + str(i)]) #loss_Wの値はparamsに依存し、self.lossの実行中にparamsがいじられることに注意
            grads["b" + str(i)] = numerical_gradient(loss_W, self.params["b" + str(i)])

        return grads

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) /float(x.shape[0])

        return(accuracy)



def numerical_gradient(f, x):
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


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return (x_train, t_train), (x_test, t_test)


if __name__ == "__main__":
    (x_train, t_train), (x_test, t_test) = get_data()
    print("Start")
