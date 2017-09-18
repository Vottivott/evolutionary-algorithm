from genetic.algorithm import GeneticAlgorithm
from genetic.crossover.single_point import SinglePointCrossover
from genetic.decoding.real_number import RealNumberDecoding
from genetic.elitism.elitism import Elitism
from genetic.initialization.real_number import RealNumberInitialization
from genetic.mutation.creep import CreepMutation
from genetic.selection.tournament_selection import TournamentSelection
from genetic.stats import average_fitness

g = lambda x: (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * x[0] ** 2 - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2))\
                 *(30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * x[0] ** 2 + 48 * x[1] - 36 * x[0] * x[1] + 27 * x[1] ** 2))
fitness_function = lambda x,generation: 1.0 / g(x)
ga = GeneticAlgorithm(30,
                      fitness_function,
                      TournamentSelection(0.7, 3),
                      SinglePointCrossover(0.8),
                      CreepMutation(0.02, 0.8, 0.04, True),
                      Elitism(1),
                      RealNumberDecoding(5),
                      RealNumberInitialization(2))
def callback(p):
    print str(p.generation) + ": " + str(p.best_fitness)
    if p.generation == 100:
        print p.best_variables
ga.run(100, callback)
# print "Average fitness over 200 runs:" + str(average_fitness(ga, 100, 200))