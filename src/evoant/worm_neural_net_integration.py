from enemy import Enemy
from neural_network import NeuralNetwork
import numpy as np

from recurrent_neural_network import RecurrentNeuralNetwork


class WormNeuralNetIntegration:
    def __init__(self, layer_sizes, input_function, output_function, concurrency, recurrent=False):
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
        self.concurrency = concurrency

    def run_network(self, copter_simulation, custom_h_layer=None):
        network_output = self.neural_network.run(self.input_function(copter_simulation), custom_h_layer)
        self.output_function(network_output, copter_simulation)

    # def clear_h(self):
    #     self.neural_network.h = self.get_empty_h()

    def initialize_h(self):
        self.neural_network.h = self.get_initial_h()

    # def set_custom_h_layer(self, h):
    #     self.neural_network.set_custom_h_layer(h)

    def set_weights_and_possibly_initial_h(self, variables):
        num_weights = self.neural_network.number_of_weights
        if len(variables) > num_weights:
            self.set_weights_and_initial_h(variables[:num_weights], np.hstack(np.array(variables[num_weights:]) for _ in range(self.concurrency)))
        else:
            print "No initial h encoded in the chromosome, initializing to all zeros."
            self.neural_network.initial_h = self.get_all_zeros_h_vector()
            self.set_weights(variables)


    def set_weights(self, weights):
        self.neural_network.set_weights_from_single_vector(weights)
        if self.recurrent:
            self.neural_network.h = [None] + [np.zeros((size, 1)) for size in self.neural_network.layer_sizes[1:-1]] # Reset state vectors
        else:
            print "NOT RECURRENT!"

    def set_weights_and_initial_h(self, weights, initial_h):
        self.neural_network.set_weights_from_single_vector(weights)
        self.neural_network.initial_h = initial_h
        # TODO: Add support for multiple layers
        if self.recurrent:
            if len(self.neural_network.layer_sizes) > 3:
                print "MORE THAN ONE HIDDEN LAYER NOT SUPPORTED YET, BUT EASY TO FIX!"
            self.neural_network.h = [None, np.copy(self.neural_network.initial_h)] # Reset state vectors
        else:
            print "NOT RECURRENT!"

    def get_all_zeros_h_vector(self):
        return np.zeros((self.neural_network.layer_sizes[1], 1))

    def get_initial_h(self):
        return [None, np.copy(self.neural_network.initial_h)]#[None] + [np.zeros(h.shape) for h in self.neural_network.h[1:-1]]


    def get_number_of_variables(self):
        return self.neural_network.get_total_number_of_weights() + self.neural_network.get_h_size()

def get_worm_neural_net_integration(worm_simulation, mirrored = False):

    num_fish = len(worm_simulation.worm.fish)

    num_ground_radars = worm_simulation.worm.fish[0].radar_system.num_front_radars + worm_simulation.worm.fish[0].radar_system.num_back_radars
    num_attr_radars = worm_simulation.worm.fish[0].radar_system.num_attr_radars

    input_layer_size = 0

    input_layer_size += 2 # velocity in up-direction + velocity in down-direction
    input_layer_size += 2 # velocity in left-direction + velocity in right-direction

    input_layer_size += num_ground_radars
    input_layer_size += num_attr_radars

    middle_layer_size = 50

    output_layer_size = 4 # output acc +x, -x, +y, -y

    layer_sizes = (input_layer_size, middle_layer_size, output_layer_size)


    def worm_input_function(sim):
        input = np.zeros((input_layer_size, num_fish))

        for i, f in enumerate(sim.worm.fish):
            yvel = np.asscalar(f.velocity[1])
            if yvel <= 0:
                velocity_up = -yvel / sim.worm.max_y_velocity
                velocity_down = 0
            else:
                velocity_up = 0
                velocity_down = yvel / sim.worm.max_y_velocity

            if mirrored:
                xvel = np.asscalar(f.velocity[0]) * -1.0
            else:
                xvel = np.asscalar(f.velocity[0])
            if xvel <= 0:
                velocity_left = -xvel / sim.worm.max_x_velocity
                velocity_right = 0
            else:
                velocity_left = 0
                velocity_right = xvel / sim.worm.max_x_velocity
            input[0:4, i:i+1] = np.array([[velocity_up], [velocity_down], [velocity_left], [velocity_right]])
            index = 4

            input[index:index + num_ground_radars, i:i+1] = np.array(
                [[radar.read(f.position, sim.level)[1]] for radar in f.radar_system.radars])
            index += num_ground_radars


            object_list = list(worm_simulation.worm.fish)
            del object_list[i]

            dist_vec, attr_vecs = f.radar_system.fish_radar.read_dist_vector_and_attribute_vectors(f.position, object_list, worm_simulation.level)
            input[index:index + num_attr_radars, i:i+1] = np.vstack([dist_vec] + attr_vecs)
            index += num_attr_radars

            assert index == input_layer_size

        return input

    def worm_output_function(network_output, sim):

        if mirrored:
            for i,f in enumerate(sim.worm.fish):
                f.velocity[0] += -1.0 * float(network_output[0, i] - network_output[1, i]) # * 1.0=acc
                f.velocity[1] += float(network_output[2, i] - network_output[3, i]) # * 1.0=acc
        else:
            for i,f in enumerate(sim.worm.fish):
                f.velocity[0] += float(network_output[0, i] - network_output[1, i]) # * 1.0=acc
                f.velocity[1] += float(network_output[2, i] - network_output[3, i]) # * 1.0=acc


    return WormNeuralNetIntegration(layer_sizes, worm_input_function, worm_output_function, concurrency=num_fish, recurrent=True)
