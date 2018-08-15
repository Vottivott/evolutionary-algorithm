import numpy as np
from itertools import count, imap, chain

# from stats import *

import random

# Fix random state for reproducability
#random_seed = 19680601
#np.random.seed(random_seed)
#random.seed(random_seed)

def extract_function(function_or_object_with_function, function_name):
    x = function_or_object_with_function
    return getattr(x, function_name, x)

class PSOPopulationData:
    """
    Data from a run of a pso algorithm, containing information about the current population
    """
    def __init__(self, generation, positions, velocities, fitness_scores, particle_best_position, particle_best_performance, swarm_best_position, swarm_best_performance, inertia_weight):
        self.generation = generation
        self.positions = positions
        self.velocities = velocities
        self.fitness_scores = fitness_scores
        self.particle_best_position = particle_best_position
        self.particle_best_performance = particle_best_performance
        self.swarm_best_position = swarm_best_position
        self.swarm_best_performance = swarm_best_performance
        self.inertia_weight = inertia_weight

class PrunedPSOPopulationData:
    """
    Data from a run of a pso algorithm, containing information about the current population
    """
    def __init__(self, population_data):
        self.generation = population_data.generation
        self.fitness_scores = population_data.fitness_scores
        self.swarm_best_position = population_data.swarm_best_position
        self.swarm_best_performance = population_data.swarm_best_performance



class ParticleSwarmOptimizationAlgorithm:
    """
    Positions/Velocities are stored as numpy column vectors,
    and the population is stored as a list of such position vectors.
    """
    def __init__(self, swarm_size, num_variables, fitness_function, x_min, x_max, v_max, alpha, delta_t, cognition, sociability, initial_inertia_weight, inertia_weight_decay, min_inertia_weight):
        """
        :param swarm_size: typically between 30 and 1000 (must be even)
        :param fitness_function: function, or object with function, evaluate(variables, generation) returning the fitness score
        :param initialization_algorithm: function, or object with function, initialize_chromosome() returning a new chromosome. If a population is to be specified to the run function, this is not needed.
        """
        self.swarm_size = swarm_size
        self.num_variables = num_variables
        self.x_min = x_min
        self.x_max = x_max
        self.v_max = v_max
        self.alpha = alpha
        self.delta_t = delta_t
        self.cognition = cognition
        self.sociability = sociability
        self.initial_inertia_weight = initial_inertia_weight
        self.inertia_weight_decay = inertia_weight_decay
        self.min_inertia_weight = min_inertia_weight
        if swarm_size % 2 == 1:
            raise ValueError('The population size must be even!')
        self.evaluate = extract_function(fitness_function, "evaluate")
        # self.initialize = extract_function(initialization_algorithm, "initialize")
        # self.update = extract_function(update_algorithm, "update")

    def run(self, num_generations=None, generation_callback=None, population_data=None):
        """
        :param num_generations: the number of generations, or None to continue indefinitely
        :param generation_callback: an optional function generation_callback(population_data) returning a boolean True to continue or False to stop.
        :param population_data: if None, a new population is initialized normally. Otherwise, the population in the specified population data is used.
        :return: an instance of PSOPopulationData with information about the final population
        """
        # Initialize population
        if population_data is None:
            positions = [np.full((self.num_variables, 1), self.x_min) + np.random.random((self.num_variables, 1)) * (self.x_max - self.x_min) for _ in range(self.swarm_size)]
            velocities = [self.alpha/self.delta_t * (-(self.x_max - self.x_min)/2.0 + np.random.random((self.num_variables, 1)) * (self.x_max - self.x_min)) for _ in range(self.swarm_size)]
            inertia_weight = self.initial_inertia_weight
            generation = 1
        else:
            positions = population_data.positions
            velocities = population_data.velocities
            inertia_weight = population_data.inertia_weight
            generation = population_data.generation
        fitness_scores = [None for _ in range(self.swarm_size)]
        particle_best_position = [None for _ in range(self.swarm_size)]
        particle_best_performance = [float('-inf') for _ in range(self.swarm_size)]
        swarm_best_position = None
        swarm_best_performance = float('-inf')


        while True:


            if population_data is not None:

                # Use stored data first time if supplied

                positions, velocities, fitness_scores, particle_best_position, particle_best_performance, swarm_best_position, swarm_best_performance, inertia_weight = self.use_population_data(population_data)
                population_data = None

            else:

                # Evaluate the particles

                for i, position in enumerate(positions):
                    fitness_scores[i] = self.evaluate(position, generation)
                    if fitness_scores[i] > particle_best_performance[i]:
                        particle_best_performance[i] = np.array(fitness_scores[i])
                        particle_best_position[i] = np.array(position)
                        if fitness_scores[i] > swarm_best_performance:
                            swarm_best_performance = np.array(fitness_scores[i])
                            swarm_best_position = np.array(position)
                            # print swarm_best_performance
                # Call optional callback function and check if finished

                data = PSOPopulationData(generation, positions, velocities, fitness_scores, particle_best_position, particle_best_performance, swarm_best_position, swarm_best_performance, inertia_weight)
                if (generation_callback is not None and generation_callback(data) is False) or generation == num_generations:
                    return data


            # Update particle velocities and positions

            # print velocities
            for i in range(self.swarm_size):
                velocities[i] = inertia_weight * velocities[i]\
                                + self.cognition * np.random.random((self.num_variables, 1)) * (particle_best_position[i] - positions[i]) / self.delta_t\
                                + self.sociability * np.random.random((self.num_variables, 1)) * (swarm_best_position - positions[i]) / self.delta_t
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)
                positions[i] += velocities[i] * self.delta_t

            generation += 1

    def use_population_data(self, population_data):
        return population_data.positions,\
        population_data.velocities,\
        population_data.fitness_scores,\
        population_data.particle_best_position,\
        population_data.particle_best_performance,\
        population_data.swarm_best_position,\
        population_data.swarm_best_performance, \
        population_data.inertia_weight





if __name__ == "__main__":

    g = lambda x: (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))\
                 *(30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    fitness_function = lambda x,generation: 1.0 / g(x)

    vars = 2

    pso = ParticleSwarmOptimizationAlgorithm(30,                # swarm_size
                                             vars,              # num_variables
                                             fitness_function,  # fitness_function
                                             -5.0, 5.0,         # x_min, x_max
                                             8.0,               # v_max
                                             0.1,               # alpha
                                             0.1,               # delta_t
                                             2.0,               # cognition
                                             2.0,               # sociability
                                             1.4,               # initial_inertia_weight
                                             0.99,              # inertia_weight_decay
                                             0.35)              # min_inertia_weight

    # import matplotlib.pyplot as plt

    num_generations = 100

    def callback(p):

        # x = []
        # y = []
        # for pos in p.positions:
        #     x.append(pos[0])
        #     y.append(pos[1])
        #
        # plt.scatter(x, y)
        # plt.show()

        if p.generation == num_generations:
            print str(p.generation) + ": " + str(p.swarm_best_performance)
            print p.swarm_best_position


    pso.run(num_generations, callback)
    # print "Average fitness over 200 runs:" + str(average_fitness(ga, 100, 200))