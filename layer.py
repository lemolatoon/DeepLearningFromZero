import numpy as np
import perceptron as pcp
import traceback


class Layer:
    def __init__(self, input_num, output_num, ws, bs, activation=None):
        self.input_num = input_num
        self.output_num = output_num
        self.ws = ws
        self.bs = bs
        if activation is not None:
            if activation == "sigmoid":
                self.activation = self.sigmoid
            elif activation == "identity":
                self.activation = self.identity_function
            elif activation == "softmax":
                self.activation = self.softmax
            else:
                try:
                    raise Exception #活性化関数の名前が不正です
                except:
                    traceback.print_exc()

    def calc(self, xs):
        a = (np.dot(xs, self.ws) + self.bs) # asは予約語なので注意
        return self.activation(a)

    def activation(a):
        return Layer.sigmoid(a)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    @staticmethod
    def identity_function(x):
        return x

    @staticmethod
    def softmax(x):
        c = np.max(x)
        exp_a = np.exp(x - c) # overflow対策
        sum_exp_a = np.sum(exp_a)
        return exp_a / sum_exp_a

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


if __name__ == "__main__":
    X = np.array([1.0, 0.5])

    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    #print(np.dot(X, W1))
    layer1 = Layer(3, W1, B1, activation="sigmoid")
    layer2 = Layer(2, W2, B2, activation="sigmoid")
    layer3 = Layer(2, W3, B3, activation="identity")
    #tmp = layer.calc(X)
    network = Network(layers=(layer1, layer2, layer3))
    y = network.calc(X)
    print(str(X) + " -> " + str(y))


    print("a")