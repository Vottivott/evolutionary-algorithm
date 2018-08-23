import numpy as np
from math import ceil
import random

class SigmoidActivationFunction:
    name = "sigmoid"
    @staticmethod
    def f(z):
        return 1.0/(1.0 + np.exp(-z))
    @staticmethod
    def derivative(z, f):
        return np.multiply(f, (1.0 - f))

activation_functions_list = [SigmoidActivationFunction]
# activation_functions = {function.name:function for function in activation_functions_list}
activation_functions = dict((function.name, function) for function in activation_functions_list) # python 2.6 support

def initial_weights(size, prev_size):
    return 0.2 * np.random.rand(size, prev_size) - 0.1

def output_activation_to_digit(a):
    return max(range(10), key=lambda x:a[x])


class NeuralNetwork:
    def __init__(self, layer_sizes, activation_function=SigmoidActivationFunction):
        # Let us use layer indices 0...L
        self.L = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.W = [None] + [initial_weights(size, prev_size) for size, prev_size in zip(layer_sizes[1:],layer_sizes)]
        self.b = [None] + [np.zeros((size,1)) for size in layer_sizes[1:]]
        self.activation_function = activation_function
        self.number_of_weights = sum(size*(prev_size+1) for size, prev_size in zip(layer_sizes[1:],layer_sizes))

    def set_weights_from_single_vector(self, vector):
        index = 0
        for layer in range(1,self.L):
            size = self.layer_sizes[layer]
            prev_size = self.layer_sizes[layer-1]
            new_index = index + size*prev_size
            self.W[layer] = vector[index:new_index].reshape((size, prev_size))
            index = new_index
            new_index = index + size
            self.b[layer] = vector[index:new_index]
            index = new_index

    def get_total_number_of_weights(self):
        return self.number_of_weights
    

    def forward_pass(self, z, a, x):
        a[0] = x
        for layer in range(1, self.L):
            # print np.dot(self.W[layer], a[layer-1]).shape
            # print self.b[layer].shape
            z[layer] = np.dot(self.W[layer], a[layer-1]) + self.b[layer]
            a[layer] = self.activation_function.f(z[layer])

    def run(self, input, custom_h_layer=None):
        z = [None] + [np.zeros((size, 1)) for size in self.layer_sizes[1:]]
        a = [np.zeros((size, 1)) for size in self.layer_sizes]
        self.forward_pass(z, a, input)
        return a[-1]

    def avg_cost(self, cost_function, training_set):
        C_total = 0.0
        for x,y in training_set:
            C_total += cost_function.f(self.run(x), y)
        return float(C_total) / len(training_set)




if __name__ == "__main__":
    nn = NeuralNetwork((784, 100, 10))
    print nn.run(np.random.rand(784,1))

