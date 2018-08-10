from enemy import Enemy
from neural_network import NeuralNetwork
import numpy as np

from recurrent_neural_network import RecurrentNeuralNetwork


class WormNeuralNetIntegration:
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
            self.set_weights_and_initial_h(variables[:num_weights], variables[num_weights:])
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

def get_worm_neural_net_integration(worm_simulation):

    ground_contact_radar_size = worm_simulation.worm_radar_system.ground_contact_radars[0].number_of_neurons
    num_ground_contact_radars = len(worm_simulation.worm_radar_system.ground_contact_radars)
    muscle_direction_radar_size = worm_simulation.worm_radar_system.muscle_direction_radars[0].number_of_neurons
    num_muscle_direction_radars = len(worm_simulation.worm_radar_system.muscle_direction_radars)

    num_balls = worm_simulation.worm.num_balls
    num_muscles = num_balls - 1

    input_layer_size = 0

    input_layer_size += 2 * num_balls # velocity in up-direction + velocity in down-direction
    input_layer_size += 2 * num_balls # velocity in left-direction + velocity in right-direction

    input_layer_size += num_muscles # real (not target) length of muscle

    input_layer_size += ground_contact_radar_size * num_ground_contact_radars
    input_layer_size += muscle_direction_radar_size * num_muscle_direction_radars

    middle_layer_size = 100

    output_layer_size = num_muscles + num_balls # target length for muscle + grippedness for balls

    layer_sizes = (input_layer_size, middle_layer_size, output_layer_size)


    def worm_input_function(sim):
        input_vectors = []
        for i, b in enumerate(sim.worm.balls):
            yvel = np.asscalar(b.velocity[1])
            if yvel <= 0:
                velocity_up = -yvel / sim.worm.max_y_velocity
                velocity_down = 0
            else:
                velocity_up = 0
                velocity_down = yvel / sim.worm.max_y_velocity
            y_velocity_inputs = np.array([[velocity_up], [velocity_down]])

            xvel = np.asscalar(b.velocity[0])
            if xvel <= 0:
                velocity_left = -xvel / sim.worm.max_x_velocity
                velocity_right = 0
            else:
                velocity_left = 0
                velocity_right = xvel / sim.worm.max_x_velocity
            x_velocity_inputs = np.array([[velocity_left], [velocity_right]])

            input_vectors.extend((y_velocity_inputs, x_velocity_inputs))

        for i, m in enumerate(sim.worm.muscles):
            real_length = np.linalg.norm(m.b2.position - m.b1.position)
            input_vectors.append(np.array([[real_length / sim.worm.max_real_muscle_length]]))

        for i, b in enumerate(sim.worm.balls):
            radar = worm_simulation.worm_radar_system.ground_contact_radars[i]
            if len(b.debug_bounces):
                bounce = b.debug_bounces[0]
                ground_contact_vec = radar.read_contact_vector_from_points(bounce.ball_center, bounce.position)
            else:
                ground_contact_vec = radar.read_contact_vector_from_points(None, None)
            input_vectors.append(ground_contact_vec)

        for i, m in enumerate(sim.worm.muscles):
            radar = worm_simulation.worm_radar_system.muscle_direction_radars[i]
            muscle_direction_vec = radar.read_contact_vector_from_points(m.b1.position, m.b2.position)
            input_vectors.append(muscle_direction_vec)


        input = np.vstack(input_vectors)
        # print input
        assert len(input) == input_layer_size
        return input

    def worm_output_function(network_output, sim):

        for muscle_index in range(num_muscles):
            extension = network_output[muscle_index]
            target = sim.worm.muscle_flex_length + extension * (sim.worm.muscle_extend_length - sim.worm.muscle_flex_length)
            sim.worm.muscles[muscle_index].target_length = target

        i_offset = num_muscles

        for ball_index in range(num_balls):
            grip_decision = network_output[i_offset + ball_index]
            if grip_decision >= 0.5:
                grippingness = 1.0
            else:
                grippingness = 0.0
            sim.worm.balls[ball_index].grippingness = grippingness

        i_offset += num_balls

    return WormNeuralNetIntegration(layer_sizes, worm_input_function, worm_output_function, recurrent=True)
