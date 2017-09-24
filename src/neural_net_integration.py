from neural_network import NeuralNetwork
import numpy as np

from recurrent_neural_network import RecurrentNeuralNetwork


class NeuralNetIntegration:
    def __init__(self, layer_sizes, input_function, output_function, recurrent=False):
        """
        :param layer_sizes: the sizes of the layers in a feed-forward neural network
        :param input_function: function that takes a copter_simulation and an enemy_index (or None for main copter) and returns the input to the network
        :param output_function: function that takes the output of the network and the copter_simulation and an enemy_index (or None for main copter) and modifies the copter_simulation accordingly
        """
        self.recurrent = recurrent
        if recurrent:
            self.neural_network = RecurrentNeuralNetwork(layer_sizes)
        else:
            self.neural_network = NeuralNetwork(layer_sizes)
        self.input_function = input_function
        self.output_function = output_function

    def run_network(self, copter_simulation, enemy_index=None):
        network_output = self.neural_network.run(self.input_function(copter_simulation, enemy_index))
        self.output_function(network_output, copter_simulation, enemy_index)

    def set_weights(self, weights):
        self.neural_network.set_weights_from_single_vector(weights)
        if self.recurrent:
            self.neural_network.h = [None] + [np.zeros(h.shape) for h in self.neural_network.h[1:-1]] # Reset state vectors
        else:
            print "NOT RECURRENT!"

    def get_empty_h(self):
        return [None] + [np.zeros(h.shape) for h in self.neural_network.h[1:-1]]

    def get_number_of_variables(self):
        return self.neural_network.get_total_number_of_weights()

def evocopter_neural_net_integration(copter_simulation):
    input_layer_size = 0
    input_layer_size += copter_simulation.radar_system.num_front_radars * 2
    input_layer_size += copter_simulation.radar_system.num_back_radars * 2
    input_layer_size += 2 # velocity in up-direction + velocity in down-direction
    input_layer_size += 2 # velocity in left-direction + velocity in right-direction

    middle_layer_size = 50

    output_layer_size = 2

    layer_sizes = (input_layer_size, middle_layer_size, output_layer_size)

    def evocopter_input_function(sim, enemy_index):
        yvel = np.asscalar(sim.copter.velocity[1])
        if yvel <= 0:
            velocity_up = -yvel / sim.copter.max_y_velocity
            velocity_down = 0
        else:
            velocity_up = 0
            velocity_down = yvel / sim.copter.max_y_velocity
        velocity_inputs = np.array([[velocity_up], [velocity_down]])

        xvel = np.asscalar(sim.copter.velocity[0])
        if xvel <= 0:
            velocity_left = -xvel / sim.copter.max_x_velocity
            velocity_right = 0
        else:
            velocity_left = 0
            velocity_right = xvel / sim.copter.max_x_velocity
        x_velocity_inputs = np.array([[velocity_left], [velocity_right]])

        dist_inputs = np.array([[radar.read(copter_simulation.copter.position, copter_simulation.level)[1]] for radar in copter_simulation.radar_system.radars])
        dist_to_enemy_inputs = np.full((len(copter_simulation.radar_system.radars),1),1.0)#np.array([[radar.read_rect(copter_simulation.copter.position, copter_simulation.level, copter_simulation.enemies)[1]] for radar in copter_simulation.radar_system.radars])

        input = np.vstack((velocity_inputs, dist_inputs, dist_to_enemy_inputs, x_velocity_inputs))
        # print input
        assert len(input) == input_layer_size
        return input

    def evocopter_output_function(network_output, copter_simulation, enemy_index):
        should_fire = network_output[0] > 0.5
        copter_simulation.copter.firing = should_fire
        if network_output[1] > 0.5:
            copter_simulation.copter_shoot()
        # print should_fire

    return NeuralNetIntegration(layer_sizes, evocopter_input_function, evocopter_output_function, recurrent=False)

