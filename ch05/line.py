import numpy as np
from network import network
from draw import *

def main():
    x = np.array([0.1, 0.2, 0.3, 0.5])
    y = np.array([0.092, 0.185, 0.279, 0.462])
    z = np.polyfit(x, y, 1)
    a, b = z

    print("a:{}, b:{}".format(a, b))
    x_reg = np.array([0, 0.6])
    y_reg = a * x_reg + b
    print(x)
    yy = a * x + b
    print(yy)
    print(y*30)
    print(yy*30)

    plt.scatter(x, y)

    plt.plot(x_reg, y_reg, color="red")
    plt.savefig("polyfit_2.png")
    plt.show()
    


if __name__ == "__main__":
    main()