import numpy as np
from evomath import *

def rotate_cw(vector):
    return np.array([[float(vector[1])], [float(-vector[0])]])

def rotate_ccw(vector):
    return np.array([[float(-vector[1])], [float(vector[0])]])

class LineSegment:
    def __init__(self, left, right, is_ceiling):
        self.left = left
        self.right = right
        self.delta = right - left
        self.tangent = normalized(self.delta)
        if is_ceiling:
            self.normal = rotate_ccw(self.tangent)
        else:
            self.normal = rotate_cw(self.tangent)


