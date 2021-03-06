from genetic.algorithm import GeneticAlgorithm
from genetic.crossover.single_point import SinglePointCrossover
from genetic.decoding.binary import BinaryDecoding
from genetic.decoding.real_number import RealNumberDecoding
from genetic.elitism.elitism import Elitism
from genetic.initialization.binary import BinaryInitialization
from genetic.initialization.real_number import RealNumberInitialization
from genetic.mutation.binary import BinaryMutation
from genetic.mutation.creep import CreepMutation
from genetic.selection.tournament import TournamentSelection
from genetic.stats import average_fitness, plot_fitness_curves

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
                      BinaryMutation(7.0/m),
                      Elitism(1),
                      BinaryDecoding(5,vars,var_size),
                      BinaryInitialization(m))
def callback(p):
    if p.generation == 100:
        print str(p.generation) + ": " + str(p.best_fitness)
        print p.best_variables
ga.run(100, callback)
print "Average fitness over 200 runs:" + str(average_fitness(ga, 100, 200))

# print "Average fitness over 200 runs:" + str(plot_fitness_curves(ga, 100, 200))