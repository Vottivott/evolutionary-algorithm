import numpy as np
import random

class SinglePointCrossover:
   def __init__(self, crossover_probability):
        """
        :param crossover_probability: typically between 0.7 and 1
        """
        self.crossover_probability = crossover_probability

   def cross(self, (a,b), generation):
        if random.random() < self.crossover_probability:
            chromosome_length = len(a)
            crossover_point = random.randint(1, chromosome_length-1)
            return (
                np.vstack((a[:crossover_point], b[crossover_point:])),
                np.vstack((b[:crossover_point], a[crossover_point:]))
            )
        else:
            return (a,b)

if __name__ == "__main__":
    a = np.array([[1], [2], [3]])
    b = np.array([[100], [200], [300]])
    c = SinglePointCrossover(0.8)
    new_a, new_b = c.cross((a,b), 1)
    print new_a
    print new_b