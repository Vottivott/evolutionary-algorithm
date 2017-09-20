from neural_network import NeuralNetwork
import numpy as np

class NeuralNetIntegration:
    def __init__(self, layer_sizes, input_function, output_function):
        """
        :param layer_sizes: the sizes of the layers in a feed-forward neural network
        :param input_function: function that takes a copter_simulation and returns the input to the network
        :param output_function: function that takes the output of the network and the copter_simulation and modifies the copter_simulation accordingly
        """
        self.neural_network = NeuralNetwork(layer_sizes)
        self.input_function = input_function
        self.output_function = output_function

    def run_network(self, copter_simulation):
        network_output = self.neural_network.run(self.input_function(copter_simulation))
        self.output_function(network_output, copter_simulation)

def evocopter_neural_net_integration(copter_simulation):
    count_length = 5 # number of steps per count
    num_counters = 5 # number of counter inputs
    input_layer_size = 0
    input_layer_size += len(copter_simulation.radar_system.radars)
    input_layer_size += 2 # velocity in up-direction + velocity in down-direction
    input_layer_size += num_counters # velocity in up-direction + velocity in down-direction

    middle_layer_size = 15

    output_layer_size = 1

    layer_sizes = (input_layer_size, middle_layer_size, output_layer_size)

    def evocopter_input_function(sim):
        yvel = sim.copter.velocity[1]
        if yvel <= 0:
            velocity_up = -yvel
            velocity_down = 0
        else:
            velocity_up = 0
            velocity_down = yvel
        velocity_inputs = np.array([[velocity_up],[velocity_down]])

        dist_inputs = np.array([[radar.read(copter_simulation.copter.position, copter_simulation.level)[1]] for radar in copter_simulation.radar_system.radars])

        count_index = (sim.timestep / count_length) % num_counters
        counters_inputs = np.zeros((input_layer_size, 1))
        counters_inputs[count_index] = 1.0

        input = np.vstack((velocity_inputs, dist_inputs, counters_inputs))
        return input

    def evocopter_output_function(network_output, copter_simulation):
        should_fire = network_output > 0.5
        copter_simulation.copter.firing = should_fire

    return NeuralNetIntegration(layer_sizes, evocopter_input_function, evocopter_output_function)