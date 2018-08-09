import numpy as np


def normalized(vector):
    length = np.dot(vector.T, vector) ** 0.5
    return vector / length

# v = np.array([[2.0],[0.0]])
# b = normalized(v)
# print v
# print b