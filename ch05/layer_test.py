import numpy as np

class MulLayer:
    def __init_(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(sself, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        
        return dx, dy


def main():
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1


    #forward
    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_fruits_layer = AddLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    price = add_fruits_layer.forward(apple_price, orange_price)
    all_price = mul_tax_layer.forward(price, tax)

    print("all_price: " + str(all_price))

    #backward
    dall_price = 1

    dprice, dtax = mul_tax_layer.backward(dall_price)
    dapple_price, dorange_price = add_fruits_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)

    print("dapple = " + str(dapple) + "   dapple_num = " + str(dapple_num) + "     dtax = " + str(dtax))
    print("dorange = " + str(dorange) + "   dorange_num = " + str(dorange_num) + "     dtax = " + str(dtax))

if __name__ == "__main__":
    main()
