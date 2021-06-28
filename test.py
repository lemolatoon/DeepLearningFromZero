import numpy as np

y = np.array([[1, 2, 3, 4, 5], [1, 3, 5, 7, 11]])

batch = 2
t = np.array([1, 4])

print(y)
print()
print(np.arange(batch))
print(t)
print(t + "a")
print(y[np.arange(batch), t]) #batchでそれぞれの行を取り出し、tで正解のラベルのみとりだす
print(y[1, 4]) #一行一列目