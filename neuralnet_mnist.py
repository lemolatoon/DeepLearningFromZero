import pickle
from layer import Layer, Network
from PIL import Image
import numpy as np
import random
import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist


def image_show(img):
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



if __name__ == "__main__":
    network = create_network()
    x_test, t_test = get_data()
    test(x_test, t_test, network)

