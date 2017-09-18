import numpy as np

class RealNumberInitialization:
    def __init__(self, chromosome_length):
        """
        :param chromosome_length: the number of genes
        """
        self.chromosome_length = chromosome_length

    def initialize_chromosome(self):
        return np.random.random((self.chromosome_length, 1))