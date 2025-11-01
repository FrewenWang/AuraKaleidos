import numpy as np

a = np.floor(10 * np.random.random((3, 4)))

print(a)

# 可以使用各种命令更改数组的形状。请注意，以下三个命令都返回一个修改后的数组，但不会更改原始数组：

a.reshape(6, 2)  # returns the array with a modified shape

print(a.reshape(6, 2))

a.T  # returns the array, transposed

print(a.T)
