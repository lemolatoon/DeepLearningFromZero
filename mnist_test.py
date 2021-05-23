import layer
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


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_test.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

print("Imageの表示テスト")
i = random.randrange(0, x_train.shape[0])
print(i)
img = x_train[i]
label = t_train[i]
print("label: " + str(label))
img = img.reshape(28, 28)
print("img.shape: " + str(img.shape))
image_show(img)

