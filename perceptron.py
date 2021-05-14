import numpy as np

class perceptron:
    def __init__(self, w1, w2, b):
        self.w1 = w1
        self.w2 = w2
        self.b = b
        print("w1:" + str(w1))
        print("w2:" + str(w2))
        print("b" + str(b))

    def run(self, x1, x2):
        if self.b + self.w1 * x1 + self.w2 * x2 <= 0: 
            print("self.w1 * x1 + self.w2 * x2")
            print(self.w1 * x1 + self.w2 * x2)
            return 0
        else:
            print("self.w1 * x1 + self.w2 * x2")
            print(self.w1 * x1 + self.w2 * x2)
            return 1

class np_perceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        #print("w:" + str(w))
        #print("b" + str(b))

    def run(self, x):
        tmp = np.sum(self.w * x) + self.b
        if  tmp <= 0: 
            #print("tmp:" + str(tmp))
            return 0
        else:
            #print("tmp:" + str(tmp))
            return 1

    def check(self):
        print(self.run(np.array([0, 0])))
        print(self.run(np.array([1, 0])))
        print(self.run(np.array([0, 1])))
        print(self.run(np.array([1, 1])))

class gate:
    def __init__(self, perceptron=None):
        if perceptron is not None:
            self.perceptron = perceptron
            self.run = perceptron.run
        else:
            AND = np_perceptron(np.array([0.5, 0.5]), -0.7)
            NAND = np_perceptron(np.array([-0.5, -0.5]), 0.7)
            OR = np_perceptron(np.array([0.5, 0.5]), -0.2)
            OR = gate(OR)
            AND = gate(AND)
            NAND = gate(NAND)
            self.AND = AND
            self.OR = OR
            self.NAND = NAND
        
    def run(self, x):
        return 0 

    def __call__(self, x):
        return self.run(x)
         
    def check(self):
            print(self.run(np.array([0, 0])))
            print(self.run(np.array([1, 0])))
            print(self.run(np.array([0, 1])))
            print(self.run(np.array([1, 1])))

class XOR(gate):
    def run(self, x):
        print("==XOR==")
        print("input:" + str(x))
        s1 = self.NAND(x)
        print("NAND:" + str(s1))
        s2 = self.OR(x)
        print("OR:" + str(s2))
        y = self.AND(np.array([s1, s2]))
        print("AND:" + str(y))
        print("==XOR==")
        return y




if __name__ == "__main__":
    xor = XOR()
    xor.check()