import numpy as np

class BinaryInitialization:
    def __init__(self, chromosome_length):
        """
        :param number_of_variables: number of variables encoded in the chromosome
        :param bits_per_variable: number of bits used to store one variable
        """
        self.chromosome_length = chromosome_length

    def initialize_chromosome(self):
        return np.random.choice([0, 1], (self.chromosome_length, 1))