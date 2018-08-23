import numpy as np
from math import ceil
import random


class SigmoidActivationFunction:
    name = "sigmoid"

    @staticmethod
    def f(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def derivative(z, f):
        return np.multiply(f, (1.0 - f))


activation_functions_list = [SigmoidActivationFunction]
# activation_functions = {function.name: function for function in activation_functions_list}
activation_functions = dict((function.name, function) for function in activation_functions_list) # python 2.6 support

def initial_weights(size, prev_size):
    return 0.2 * np.random.rand(size, prev_size) - 0.1


def output_activation_to_digit(a):
    return max(range(10), key=lambda x: a[x])


class RecurrentNeuralNetwork:
    def __init__(self, layer_sizes, activation_function=SigmoidActivationFunction):
        # Let us use layer indices 0...L
        self.L = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.W = [None] + [initial_weights(size, prev_size) for size, prev_size in zip(layer_sizes[1:], layer_sizes)]
        self.b = [None] + [np.zeros((size, 1)) for size in layer_sizes[1:]]

        # weights from the same layer to itself
        self.W_recurrent = [None] + [initial_weights(size, size) for size in layer_sizes[1:-1]]
        self.h = [None] + [np.zeros((size, 1)) for size in layer_sizes[1:-1]] # state vectors
        # print map(lambda x:x.shape if x!=None else 0,self.h)

        self.initial_h = None # the initial state vector, ONLY THE VECTOR, NOT A LIST LIKE self.h ^

        self.activation_function = activation_function
        self.number_of_weights = sum(size * (prev_size + 1) for size, prev_size in zip(layer_sizes[1:], layer_sizes))
        self.number_of_weights += sum(size * size for size in layer_sizes[1:-1]) # recurrent weights


    def set_weights_from_single_vector(self, vector):
        index = 0
        for layer in range(1, self.L):
            size = self.layer_sizes[layer]
            prev_size = self.layer_sizes[layer - 1]
            new_index = index + size * prev_size
            self.W[layer] = vector[index:new_index].reshape((size, prev_size))
            index = new_index
            new_index = index + size
            self.b[layer] = vector[index:new_index]
            index = new_index
        for layer in range(1, self.L-1): # recurrent weights
            size = self.layer_sizes[layer]
            new_index = index + size * size
            self.W_recurrent[layer] = vector[index:new_index].reshape((size, size))
            index = new_index

    def get_total_number_of_weights(self):
        return self.number_of_weights

    def get_h_size(self):
        return self.layer_sizes[1]

    # def set_custom_h_layer(self, h):
    #     self.h = h

    def forward_pass(self, z, a, x):
        a[0] = x
        for layer in range(1, self.L-1):
            # print np.dot(self.W[layer], a[layer-1]).shape
            z[layer] = np.dot(self.W[layer], a[layer - 1]) + self.b[layer]\
                     + np.dot(self.W_recurrent[layer], self.h[layer]) # recurrent connection
            self.h[layer] = a[layer] = self.activation_function.f(z[layer])
        # output layer (no recurrency)
        z[-1] = np.dot(self.W[-1], a[-2]) + self.b[-1]
        a[-1] = self.activation_function.f(z[-1])

    def forward_pass_custom_h(self, z, a, x, custom_h):
        a[0] = x
        for layer in range(1, self.L-1):
            # print np.dot(self.W[layer], a[layer-1]).shape
            z[layer] = np.dot(self.W[layer], a[layer - 1]) + self.b[layer]\
                     + np.dot(self.W_recurrent[layer], custom_h[layer]) # recurrent connection
            custom_h[layer] = a[layer] = self.activation_function.f(z[layer])
        # output layer (no recurrency)
        z[-1] = np.dot(self.W[-1], a[-2]) + self.b[-1]
        a[-1] = self.activation_function.f(z[-1])

    def run(self, input, custom_h_layer=None):
        z = [None] + [np.zeros((size, 1)) for size in self.layer_sizes[1:]]
        a = [np.zeros((size, 1)) for size in self.layer_sizes]
        if custom_h_layer:
            self.forward_pass_custom_h(z, a, input, custom_h_layer)
        else:
            self.forward_pass(z, a, input)
        return a[-1]

    def avg_cost(self, cost_function, training_set):
        C_total = 0.0
        for x, y in training_set:
            C_total += cost_function.f(self.run(x), y)
        return float(C_total) / len(training_set)


if __name__ == "__main__":
    nn = RecurrentNeuralNetwork((784, 100, 10))
    for i in range(10):
        print i
        print nn.run(np.random.rand(784, 1))

