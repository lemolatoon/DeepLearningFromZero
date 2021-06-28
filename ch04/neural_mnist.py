import pickle
from PIL import Image
import numpy as np
import random
import sys
import os
import pandas as pd
sys.path.append(os.pardir)
from ch03.layer import Layer, Network
from dataset.mnist import load_mnist


def image_show(img):
    #print(img.shape)
    if img.shape[0] > 28:
        img = img.reshape(28, 28)
        #print("reshapeしたよ")
        #print(img.shape)
    if np.sum(img) <= img.size:
        img = img * 255.0
        #print("正規化戻したよ")
    pil_image = Image.fromarray(np.uint8(img))
    pil_image.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True, one_hot_label=False)
    return x_test, t_test
    #return x_train, t_train


def get_parameter():
    with open("sample_weight.pkl", "rb") as f:
        parameters = pickle.load(f)
    return parameters


def create_network():
    parameters = get_parameter()
    W1, W2, W3 = parameters["W1"], parameters["W2"], parameters["W3"]
    b1, b2, b3 = parameters["b1"], parameters["b2"], parameters["b3"]

    layer1 = Layer(784, 50, W1, b1, activation="sigmoid")
    layer2 = Layer(50, 100, W2, b2, activation="sigmoid")
    layer3 = Layer(100, 10, W3, b3, activation="softmax")
    network = Network((layer1, layer2, layer3))

    return network


def test(x_test, t_test, network, batch_size=100):
        i = random.randrange(0, t_test.shape[0])
        img = x_test[i]
        label = t_test[i]

        img = img.reshape(28, 28)

        image_show(img)

        normalized = x_test.astype(np.int32) / 255.0

        y = network.predict(normalized[i])
        print(y)
        y = np.argmax(y)

        print("predict: " + str(y))
        print("label: " + str(label))

        count = 0
        for i in range(0, len(t_test), batch_size):
            x_batch = normalized[i:i+batch_size]
            y_batch = network.predict(x_batch)
            #print(y_batch)
            p = np.argmax(y_batch, axis=1)
            count += np.sum(p == t_test[i:i+batch_size])
        print()
        print(count)
        print("Accuracy" + str(float(count) / len(t_test)))

def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    #param = pd.read_pickle("param_tmp.pkl")
    param = pd.read_pickle("../saved/param_tmp_1.pkl")
    w1 = param["W1"]
    w2 = param["W2"]
    b1 = param["b1"]
    b2 = param["b2"]

    layer1 = Layer(784, 100, w1, b1, activation="sigmoid")
    layer2 = Layer(100, 10, w2, b2, activation="softmax")

    net = Network((layer1, layer2))

    t = np.argmax(t_test, axis=1)
    while True:
        txt = input()
        if txt == "exit":
            sys.exit(0)
        elif txt == "acc":
            print("train acc : " + str(net.accuracy(x_train, t_train)))
            print("test acc : " + str(net.accuracy(x_test, t_test)))
        elif txt == "reload":
            path = input()
            if os.path.exists(path):
                param = pd.read_pickle(path)
            else:
                param = pd.read_pickle("param_tmp.pkl")
            w1 = param["W1"]
            w2 = param["W2"]
            b1 = param["b1"]
            b2 = param["b2"]

            net.layers[0].ws = w1
            net.layers[0].bs = b1
            net.layers[1].ws = w2
            net.layers[1].bs = b2
        else:
            i = np.random.choice(x_test.shape[0])

            x = x_test[i]
            y = net.predict(x)
            image_show(x)
            print("t : " + str(t[i]) + "        y : " + str(np.argmax(y)))



if __name__ == "__main__":
    main()
    #network = create_network()
    x_test, t_test = get_data()
    #print(x_test.shape)
    #test(x_test, t_test, network)

    network = get_parameter()
    network = pd.read_pickle("param_tmp.pkl")
    w1 = network["W1"]
    w2 = network["W2"]
    #w3 = network["W3"]
    b1 = network["b1"]
    b2 = network["b2"]
    #b3 = network["b3"]

    #network = pd.readpickle("param.pkl")

    

    xs, ts = get_data()
    i = random.randrange(0, 10000)
    x = xs[i]
    t = ts[i]
    image_show(x.reshape(28, 28))

    print(x.shape)

    x = np.dot(x,w1) + b1
    x = Layer.sigmoid(x)
    print(x.shape)
    

    x = np.dot(x,w2) + b2
    #x = Layer.sigmoid(x)
    x = Layer.softmax(x)
    print(x.shape)

    '''
    x = np.dot(x,w3) + b3
    x = Layer.softmax(x)
    '''
    print(x.shape)
    print(x)
    print(np.argmax(x))
