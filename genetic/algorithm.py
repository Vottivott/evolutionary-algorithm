import numpy as np
from itertools import count, imap, chain

from genetic.crossover.single_point import SinglePointCrossover
from genetic.decoding.binary import BinaryDecoding
from genetic.decoding.real_number import RealNumberDecoding
from genetic.elitism.elitism import Elitism
from genetic.initialization.binary import BinaryInitialization
from genetic.initialization.real_number import RealNumberInitialization
from genetic.mutation.binary import BinaryMutation
from genetic.mutation.creep import CreepMutation
from genetic.selection.tournament import TournamentSelection
# from stats import *

import random
import time

# Fix random state for reproducability
#random_seed = 19680601
#np.random.seed(random_seed)
#random.seed(random_seed)
from temp_data_io import save_temp_data, save_temp_fitness, wait_for_temp_fitness_scores, \
    load_temp_fitness_scores, clear_temp_folder


def extract_function(function_or_object_with_function, function_name):
    x = function_or_object_with_function
    return getattr(x, function_name, x)

class PopulationData:
    """
    Data from a run of a genetic algorithm, containing information about the current population
    """
    def __init__(self, generation, population, decoded_variable_vectors, fitness_scores, best_individual_index):
        self.generation = generation
        self.population = population
        self.decoded_variable_vectors = decoded_variable_vectors
        self.fitness_scores = fitness_scores
        self.best_individual_index = best_individual_index
        self.best_individual_genes = population[best_individual_index]
        self.best_variables = decoded_variable_vectors[self.best_individual_index]
        self.best_fitness = fitness_scores[best_individual_index]

class PrunedPopulationData:
    """
    Data from a run of a genetic algorithm, containing information about the current population
    """
    def __init__(self, population_data):
        self.generation = population_data.generation
        self.fitness_scores = population_data.fitness_scores
        self.best_individual_index = population_data.best_individual_index
        self.best_variables = population_data.best_variables
        self.best_fitness = population_data.best_fitness




class GeneticAlgorithm:
    """
    Chromosomes are stored as numpy column vectors,
    and the population is stored as a list of chromosomes.

    The crossover algorithm is itself responsible
    for the decision of whether crossover should take place,
    for instance with a specified crossover_probability,
    and otherwise simply returns (copies of) the chromosomes in the pair.
    The same is true for the mutation algorithm.


    """
    def __init__(self, population_size, fitness_function, selection_algorithm, crossover_algorithm, mutation_algorithm, elitism_algorithm, decoding_algorithm, initialization_algorithm=None):
        """
        :param population_size: typically between 30 and 1000 (must be even)
        :param fitness_function: function, or object with function, evaluate(variables, generation) returning the fitness score
        :param selection_algorithm: function, or object with function, select(fitness_scores, generation) returning the selected chromosome
        :param crossover_algorithm: function, or object with function, cross(pair, generation) returning the resulting crossed pair (if unchanged, return COPIES, not the original vectors)
        :param mutation_algorithm: function, or object with function, mutate(chromosome, generation) modifying the chromosome in-place
        :param elitism_algorithm: function, or object with function, elitism(population, best_individual, generation) modifying the population in-place
        :param decoding_algorithm: function, or object with function, decode(chromosome) returning a column vector of variable values
        :param initialization_algorithm: function, or object with function, initialize_chromosome() returning a new chromosome. If a population is to be specified to the run function, this is not needed.
        """
        self.population_size = population_size
        if population_size % 2 == 1:
            raise ValueError('The population size must be even!')
        self.evaluate = extract_function(fitness_function, "evaluate")
        self.select = extract_function(selection_algorithm, "select")
        self.cross = extract_function(crossover_algorithm, "cross")
        self.mutate = extract_function(mutation_algorithm, "mutate")
        self.elitism = extract_function(elitism_algorithm, "elitism")
        self.decode = extract_function(decoding_algorithm, "decode")
        self.initialize_chromosome = extract_function(initialization_algorithm, "initialize_chromosome")

    def multiprocess_evaluations(self, subfolder_name, decoded_variable_vectors, generation, num_processes, multiprocess_index):
        t0 = time.time()
        for individual_index in range(multiprocess_index, self.population_size, num_processes):
            individual_fitness = self.evaluate(decoded_variable_vectors[individual_index], generation)
            save_temp_fitness(subfolder_name, individual_index, individual_fitness)
        wait_for_temp_fitness_scores(subfolder_name, self.population_size)
        fitness_scores = load_temp_fitness_scores(subfolder_name, self.population_size)
        while 1:
            try:
                clear_temp_folder(subfolder_name)
                break
            except WindowsError:
                print "WindowsError in clear_temp_folder()"
            time.sleep(0.2)
        print "The fitness scores were evaluated in " + str(time.time() - t0) + " seconds."
        return fitness_scores

    def run(self, num_generations=None, generation_callback=None, population_data=None, multiprocess_num_processes=1, multiprocess_index=None, subfolder_name=None):
        """
        :param num_generations: the number of generations, or None to continue indefinitely
        :param generation_callback: an optional function generation_callback(population_data) returning a boolean True to continue or False to stop.
        :param population_data: if None, a new population is initialized using the specified initialization_algorithm. Otherwise, the population in the specified population data is used.
        :return: an instance of PopulationData with information about the final population
        """
        # Initialize population
        if population_data is None:
            population = [self.initialize_chromosome() for i in range(self.population_size)]
            generation = 1
        else:
            population = population_data.population
            generation = population_data.generation

        if multiprocess_num_processes > 1:
            clear_temp_folder(subfolder_name)

        while True:


            if population_data is not None:

                # Use stored data first time if supplied

                decoded_variable_vectors, fitness_scores, best_individual_index, best_individual = self.use_population_data(population_data)
                population_data = None

            else:

                # Evaluate population

                decoded_variable_vectors = map(self.decode, population)
                if multiprocess_num_processes == 1:
                    fitness_scores = [self.evaluate(vector, generation) for vector in decoded_variable_vectors]
                else:
                    save_temp_data(subfolder_name, "generation_and_decoded_variable_vectors", (generation, decoded_variable_vectors))
                    fitness_scores = self.multiprocess_evaluations(subfolder_name, decoded_variable_vectors, generation, multiprocess_num_processes, multiprocess_index)

                best_individual_index = max(xrange(len(fitness_scores)), key=fitness_scores.__getitem__)
                best_individual = np.copy(population[best_individual_index])


                # Call optional callback function and check if finished

                data = PopulationData(generation, population, decoded_variable_vectors, fitness_scores, best_individual_index)
                if (generation_callback is not None and generation_callback(data) is False) or generation == num_generations:
                    return data



            # Form the next generation

            selected_pairs_indices = ([self.select(fitness_scores, generation) for _ in range(2)] for i in range(len(population)/2))
            selected_pairs = (map(population.__getitem__, pair) for pair in selected_pairs_indices)
            crossed_pairs = (self.cross(pair, generation) for pair in selected_pairs)
            population = list(chain.from_iterable(crossed_pairs))
            for chromosome in population:
                self.mutate(chromosome, generation)
            self.elitism(population, best_individual, generation)
            generation += 1

    def use_population_data(self, population_data):
        return population_data.decoded_variable_vectors,\
        population_data.fitness_scores,\
        population_data.best_individual_index,\
        np.copy(population_data.best_individual_genes)





if __name__ == "__main__":

    g = lambda x: (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))\
                 *(30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
    fitness_function = lambda x,generation: 1.0 / g(x)

    vars = 2
    var_size = 30
    m = vars * var_size

    ga = GeneticAlgorithm(30,
                          fitness_function,
                          TournamentSelection(0.75, 3),
                          SinglePointCrossover(0.9),
                          BinaryMutation(7.0 / m),
                          Elitism(1),
                          BinaryDecoding(5, vars, var_size),
                          BinaryInitialization(m))


    def callback(p):
        if p.generation == 100:
            print str(p.generation) + ": " + str(p.best_fitness)
            print p.best_variables


    ga.run(100, callback)
    # print "Average fitness over 200 runs:" + str(average_fitness(ga, 100, 200))