import numpy as np


def normalized(vector):
    length = np.dot(vector.T, vector) ** 0.5
    return vector / length

def mean(list):
    return sum(list) / float(len(list))

# v = np.array([[2.0],[0.0]])
# b = normalized(v)
# print v
# print b