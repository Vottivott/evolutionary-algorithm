import numbers

class RealNumberDecoding:
    def __init__(self, variable_range):
        """
        :param variable_range: a tuple of (min, max) or a single number r resulting in (-r, r)
        """
        if isinstance(variable_range, numbers.Number):
            self.variable_range = (-variable_range, variable_range)
        else:
            self.variable_range = variable_range
        self.range_width = self.variable_range[1] - self.variable_range[0]

    def decode(self, chromosome):
        return self.variable_range[0] + chromosome*self.range_width




if __name__ == "__main__":
    import numpy as np
    d = RealNumberDecoding((3,5))
    print d.decode(np.array([[1],[0.5],[0]]))