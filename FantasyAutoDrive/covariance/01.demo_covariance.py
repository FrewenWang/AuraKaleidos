import numpy as np

x1 = [-2.1, -1, 4.3]
x2 = [3, 1.1, 0.12]

X = np.stack((x1, x2), axis=0)  # 每一行作为一个变量
print("np.stack((x1, x2) = \n ", np.cov(X))
# [out]:array([[11.71      , -4.286     ],
#       	  [-4.286     ,  2.144133]])
print("\n")

print("np.cov(x1, x2) = \n ", np.cov(x1, x2))
# [out]:array([[11.71      , -4.286     ],
#       	  [-4.286     ,  2.144133]])

print("\n")

print("np.cov(x1) = \n ", np.cov(x1))
# [out]:array(11.71)
