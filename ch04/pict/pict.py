from PIL import Image
import numpy as np

def inversion(im):
    im_array = np.array(im)
    if im_array[0][0] == 255:
        im_array = 255 - im_array
    im = im.reshape(28, 28)
    im = Image.fromarray(im_array)
    return im


if __name__ == "__main__":
    im = Image.open("num1.png").convert("L")
    for i in range(10):
        im = Image.open("num{}.png".format(i)).convert("L")
       
        im = inversion(im)
        im.save("num{}.png".format(i))