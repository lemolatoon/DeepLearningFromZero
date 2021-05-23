import numpy as np

def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    delta = 1e-7
    #return -np.sum(t * np.log(y + delta))

if __name__ == "__main__":
    t = 2
    t_hot = np.zeros(10)
    t_hot[t] = 1
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    y = np.array(y)
    print(sum_squared_error(y, t_hot))
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    y = np.array(y)
    print(sum_squared_error(y, t_hot))