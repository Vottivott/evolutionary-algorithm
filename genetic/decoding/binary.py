import numpy as np
import numbers

class BinaryDecoding:
    def __init__(self, variable_range, number_of_variables, bits_per_variable):
        """
        :param variable_range: a tuple of (min, max) or a single number r resulting in (-r, r)
        :param number_of_variables: number of variables encoded in the chromosome
        :param bits_per_variable: number of bits used to store one variable
        """
        if isinstance(variable_range, numbers.Number):
            self.variable_range = (-variable_range, variable_range)
        else:
            self.variable_range = variable_range
        self.range_width = self.variable_range[1] - self.variable_range[0]
        self.number_of_variables = number_of_variables
        self.bits_per_variable = bits_per_variable

    def decode(self, chromosome):
        max_raw_value = 1 - 2.0**(-self.bits_per_variable)
        negative_powers_of_two = 2.0 ** -np.arange(1,self.bits_per_variable+1).reshape(-1,1)
        reshaped_chromosome = chromosome.reshape((self.number_of_variables, self.bits_per_variable))
        raw_values = np.dot(reshaped_chromosome, negative_powers_of_two)
        normalized_values = raw_values / max_raw_value
        rescaled_values = self.variable_range[0] + normalized_values * self.range_width
        return rescaled_values


if __name__ == "__main__":
    import numpy as np
    d = BinaryDecoding(3, 2, 3)
    print d.decode(np.array([[1],[0],[1],[1],[1],[0]])) # -> [1.29, 2.14]