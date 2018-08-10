import numpy as np

from genetic.decoding.real_number import RealNumberDecoding
from genetic.initialization.real_number import RealNumberInitialization
from worm_radar_system import WormRadarSystem
from worm import Worm
from enemy import Enemy
from genetic.algorithm import GeneticAlgorithm
from genetic.crossover.single_point import SinglePointCrossover
from genetic.decoding.binary import BinaryDecoding
from genetic.elitism.elitism import Elitism
from genetic.initialization.binary import BinaryInitialization
from genetic.mutation.binary import BinaryMutation
from genetic.mutation.creep import CreepMutation
from genetic.selection.tournament import TournamentSelection
from worm_graphics import WormGraphics
from bar_level import generate_bar_level
from neural_net_integration import evocopter_neural_net_integration, black_neural_net_integration
from population_data_io import save_population_data, load_population_data
from radar_system import RadarSystem, EnemysRadarSystem
from score_colors import get_color_from_score
from shot import Shot
from smoke import Smoke

from worm_neural_net_integration import get_worm_neural_net_integration

import sys


enemy_mode = True
view_offset = 1200 / 7.0
enemy_view_offset = 6.0 * 1200 / 7.0
base_start_x = 1200
enemy_width = 20
start_x = base_start_x + view_offset
min_x = base_start_x+view_offset+5*enemy_width

ball_radius = 10.0
segment_size = 13.0#17.0
num_segments = 6
ball_ball_friction = 0.0#0.4
ball_ground_friction = 0.4
ball_mass = 10.0
spring_constant = 30.0


class WormSimulation:
    def __init__(self, level, worm):
        self.level = level
        self.worm = worm
        self.worm_radar_system = WormRadarSystem(worm.num_balls-1)
        self.gravity = np.array([[0.0],[0.4*9.8]])
        self.delta_t = 1.0/4
        self.graphics = None
        self.score = 0.0
        self.time_since_improvement = 0
        self.worm_neural_net_integration = None

    def termination_condition(self):
        return self.time_since_improvement > 600 #0

    def run(self, graphics=None):
        self.graphics = graphics
        self.timestep = 0
        self.score = 0.0
        self.time_since_improvement = 0

        while not self.termination_condition():

            if self.worm_neural_net_integration is not None:
                self.worm_neural_net_integration.run_network(self)

            self.worm.step(self.level, self.gravity, self.delta_t)

            if self.graphics:
                space, enter, ctrl = self.graphics.update(self)
                self.worm.muscles[0].target_length = space and self.worm.muscle_flex_length or self.worm.muscle_extend_length
                self.worm.balls[-1].grippingness = ctrl
                if not ctrl:
                    self.worm.balls[-1].gripping = False
                if enter:
                    return

            potential_score = self.worm.get_distance_travelled()
            if potential_score > self.score:
                self.time_since_improvement = 0
                self.score = potential_score
            else:
                self.time_since_improvement += 1
            self.timestep += 1







def run_evaluation(level, fitness_calculator, use_graphics=False):
    s.level = level
    s.worm_neural_net_integration = worm_neural_net_integration
    if s.worm_neural_net_integration is not None:
        s.worm_neural_net_integration.initialize_h()
    s.worm = Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_friction, ball_ground_friction, ball_mass, spring_constant)
    s.run(graphics if use_graphics else None)  # - start_x  # use moved distance from start point as fitness score
    # if watch_only:
    #     print fitness
    return fitness_calculator(s)

def run_evaluations(levels, fitness_calculator, use_graphics=False):
    fitness_total = 0.0
    for level in levels:
        fitness_total += run_evaluation(level, fitness_calculator, use_graphics)
    return fitness_total / num_levels


def run_worm_evaluation(variables, use_graphics=False):
    worm_neural_net_integration.set_weights_and_possibly_initial_h(variables)
    def fitness_calculator(sim):
        return sim.score
    return run_evaluations(levels, fitness_calculator, use_graphics)



def generate_levels(close_end=True):
    result = []
    for i in range(num_levels):
        level = generate_bar_level(level_length, close_end)
        result.append(level)
    return result


class WormFitnessFunction:
    def __init__(self):
        self.last_generation = -1
        self.debug_ind_n = 1

    def evaluate(self, variables, generation):

        if generation != self.last_generation:
            # if self.last_generation == -1: # TEST: ONLY ON PROGRAM START
                # load_latest_worm_network()
            self.last_generation = generation
            global levels
            levels = generate_levels()
            self.debug_ind_n = 1
        fitness = run_worm_evaluation(variables, False)
        print get_color_from_score(fitness, False) + str(int(fitness)),
        #print "("+str(self.debug_ind_n) + "): " + str(fitness)
        self.debug_ind_n += 1
        return fitness


worm_subfoldername = "worm_a"


def run_evolution_on_worm():

    # enemy_population_data = load_population_data(enemy_subfoldername, -1)
    # enemy_neural_net_integration.set_weights_and_possibly_initial_h(enemy_population_data.best_variables)
    # load_latest_enemy_network()

    vars = worm_neural_net_integration.get_number_of_variables()


    # var_size = 30
    # m = vars * var_size

    ga = GeneticAlgorithm(100,
                          WormFitnessFunction(),
                          TournamentSelection(0.75, 3),
                          SinglePointCrossover(0.75),#0.9),
                          CreepMutation(1.0 / vars, 0.8, 0.005, True), #BinaryMutation(2.0 / m),
                          Elitism(1),
                          RealNumberDecoding(5.0), #BinaryDecoding(5, vars, var_size),
                          RealNumberInitialization(vars))# BinaryInitialization(m))

    def worm_callback(p, watch_only=False):

        # # if p.generation == 100:
        if not watch_only:
            #if p.generation % 50 == 0: # Save only every 50th generation
            save_population_data(worm_subfoldername, p, keep_last_n=10, keep_mod = None)
        average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
        print "\n[ " + str(p.generation) + ": " + str(
            p.best_fitness) + " : " + str(
            average_fitness) + " ]\n"
        if watch_only or (graphics is not None):# and p.generation % 10 == 0):
            fitness = run_worm_evaluation(p.best_variables, True)
            print "Fitness: " + str(fitness)

    watch_only = False
    global worm_population_data
    worm_population_data = load_population_data(worm_subfoldername, -1)
    # # g = worm_population_data.best_individual_genes

    if True:
        if watch_only:
            while 1:
                # BinaryMutation(100.0 / m).mutate(g, 1)
                # worm_population_data.best_variables = BinaryDecoding(5, vars, var_size).decode(g)
                global levels
                levels = generate_levels()
                worm_callback(worm_population_data, True)
                # watch_run(worm_population_data)
        else:
            ga.run(None, worm_callback, population_data=worm_population_data)

def watch_best_worm():
    global worm_population_data
    worm_population_data = load_population_data(worm_subfoldername, -1)
    global levels
    levels = generate_levels()
    fitness = run_worm_evaluation(worm_population_data.best_variables, True)
    print "Fitness: " + str(fitness)

def load_latest_worm_network():
    worm_population_data = load_population_data(worm_subfoldername, -1)
    global worm_neural_net_integration
    worm_neural_net_integration.set_weights_and_possibly_initial_h(worm_population_data.best_variables)

graphics = WormGraphics()

levels = []


num_levels = 7
level_length = 10000


new_level = generate_bar_level(5000)
s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_friction, ball_ground_friction, ball_mass, spring_constant))
worm_neural_net_integration = get_worm_neural_net_integration(s)
s.worm_neural_net_integration = worm_neural_net_integration


run_evolution_on_worm()
# watch_best_worm()


# while 1:
#
#
#
#     new_level = generate_bar_level(5000)
#     s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_friction, ball_ground_friction, ball_mass, spring_constant))
#
#     worm_neural_net_integration = get_worm_neural_net_integration(s)
#
#
#     s.run(graphics)