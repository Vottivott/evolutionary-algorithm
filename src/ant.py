from rectangular import Rectangular
import numpy as np

class Ant(Rectangular):
    def __init__(self, position, size):
        Rectangular.__init__(self, position, size, size)

    def step(self, delta_time):
        return True





