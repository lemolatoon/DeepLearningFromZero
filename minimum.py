import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def f(x):
    return x[0] * np.sin(x)
    #return 0.01*x*x + 0.1*x
    #return np.sin(x)

def numerical_diff(f, x):
    '''
    h = 1e-7
    return (f(x+h) - f(x-h)) / (2*h)
    '''

    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)

        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp

    return grad


#x = np.arange(0, 20, 0.1)
x = np.array([np.arange(0, 20, 0.1), np.arange(0, 20, 0.1)])
y = f(x)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
sys.exit(0)

t = 3
if len(sys.argv) == 2:
    t = float(sys.argv[1])
print(sys.argv)
print(t)

a = numerical_diff(f, t)
y2 = a * (x - t) + f(t)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

if t > 20:
    sys.exit(0)

os.system("python c:/Users/khish/MyFolder/private/DeepLearningFromZero/minimum.py " + str(t + 0.01))
if t <= 20:
    sys.exit(0)