import numpy as np

from pso_stats_handler import PSOStatsHandler

from mail import send_mail_message_with_image
from genetic.decoding.real_number import RealNumberDecoding
from genetic.initialization.real_number import RealNumberInitialization
from pso.algorithm import ParticleSwarmOptimizationAlgorithm
from stats_data_io import save_stats, load_stats
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
from bar_level import generate_bar_level, generate_planar_bar_level
from neural_net_integration import evocopter_neural_net_integration, black_neural_net_integration
from population_data_io import save_population_data, load_population_data
from radar_system import RadarSystem, EnemysRadarSystem
from score_colors import get_color_from_score
from shot import Shot
from smoke import Smoke

import pygame

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
num_segments = 6#3 #6 # worm_b=6
ball_ball_restitution = 0.0#0.4
ball_ground_restitution = 0.7
ball_ground_friction = 0.4
ball_mass = 10.0
spring_constant = 30.0


class WormSimulation:
    def __init__(self, level, worm):
        self.level = level
        self.worm = worm
        self.worm_radar_system = WormRadarSystem(worm.num_balls-1)
        self.gravity = np.array([[0.0],[0.6*9.8]])
        self.delta_t = 1.0/4
        self.graphics = None
        self.score = 0.0
        self.time_since_improvement = 0
        self.worm_neural_net_integration = None

    def termination_condition(self):
        # return self.time_since_improvement > 600 #0
        # return self.timestep > 300
        # return (not (self.graphics and self.graphics.user_control)) and self.timestep > 150
        # return (not (self.graphics and self.graphics.user_control)) and self.timestep > 200
        return (not (self.graphics and self.graphics.user_control)) and self.timestep > 600
        # return False

    def run(self, graphics=None):
        self.graphics = graphics
        self.timestep = 0
        self.score = 0.0
        self.time_since_improvement = 0

        while not self.termination_condition():

            if self.worm_neural_net_integration is not None:
                self.worm_neural_net_integration.run_network(self)
            elif self.graphics is not None and self.graphics.user_control:
                if self.graphics.keys is not None:
                    key_names = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0]
                    keys = [self.graphics.keys[k] for k in key_names]
                    for i,m in enumerate(self.worm.muscles):
                        m.target_length = keys[i] and self.worm.muscle_flex_length or self.worm.muscle_extend_length
                    for i, b in enumerate(self.worm.balls):
                        b.grippingness = keys[self.worm.num_balls-1 + i] and 1.0 or 0.0


            self.worm.step(self.level, self.gravity, self.delta_t)

            if self.graphics:
                space, enter, ctrl = self.graphics.update(self)
                # self.worm.muscles[0].target_length = space and self.worm.muscle_flex_length or self.worm.muscle_extend_length
                # self.worm.balls[-1].grippingness = ctrl
                # if not ctrl:
                #     self.worm.balls[-1].gripping = False
                if enter:
                    return

            potential_score = self.worm.get_distance_travelled()
            if potential_score > self.score:
                self.time_since_improvement = 0
                self.score = np.copy(potential_score)
            else:
                self.time_since_improvement += 1
            self.timestep += 1







def run_evaluation(level, fitness_calculator, use_graphics=False):
    s.level = level
    s.worm_neural_net_integration = worm_neural_net_integration
    if s.worm_neural_net_integration is not None:
        s.worm_neural_net_integration.initialize_h()
    s.worm = Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant)
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
        # level = generate_bar_level(level_length, close_end)
        level = generate_planar_bar_level(level_length, close_end)
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


# worm_subfoldername = "worm_b"
# worm_subfoldername = "worm_2segs_planar"
# worm_subfoldername = "PSO_worm_3segs_planar"
worm_subfoldername = "PSO_worm_6segs_planar"


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
        # if watch_only or (graphics is not None):# and p.generation % 10 == 0):
        #     fitness = run_worm_evaluation(p.best_variables, True)
        #     print "Fitness: " + str(fitness)

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

def run_pso_on_worm():
    num_vars = worm_neural_net_integration.get_number_of_variables()

    pso = ParticleSwarmOptimizationAlgorithm(30,#30,  # swarm_size
                                             num_vars,  # num_variables
                                             WormFitnessFunction(),  # fitness_function
                                             -1.5, 1.5,  # x_min, x_max
                                             3.0,#8.0,  # v_max
                                             0.3,  # alpha
                                             1.0,  # delta_t
                                             2.0,  # cognition
                                             2.0,  # sociability
                                             1.4,  # initial_inertia_weight
                                             0.995,  # inertia_weight_decay
                                             0.35)  # min_inertia_weight

    def worm_callback(p, watch_only=False):

        # # if p.generation == 100:
        if not watch_only:
            #if p.generation % 50 == 0: # Save only every 50th generation
            save_population_data(worm_subfoldername, p, keep_last_n=10, keep_mod = None)
            save_stats(worm_subfoldername, stats_handler, p)
            stats = load_stats(worm_subfoldername)
            if stats is not None and (len(stats["best_fitness"]) < 2 or stats["best_fitness"][-1] != stats["best_fitness"][-2]):
                stats_handler.produce_graph(stats, worm_subfoldername + ".png")
                msg = str(float(p.best_fitness)) + "\ngeneration " + str(p.generation)
                send_mail_message_with_image(worm_subfoldername, msg, worm_subfoldername + ".png", worm_subfoldername)

        average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
        print "\n[ " + str(p.generation) + ": " + str(
            p.best_fitness) + " : " + str(
            average_fitness) + " ]     inertia = " + str(p.inertia_weight) + "\n"
        # if watch_only or (graphics is not None):# and p.generation % 10 == 0):
        #     fitness = run_worm_evaluation(p.best_variables, True)
        #     print "Fitness: " + str(fitness)

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
            pso.run(None, worm_callback, population_data=worm_population_data)


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

stats_handler = PSOStatsHandler()

# graphics = WormGraphics()

levels = []


num_levels = 15#4#30#15
level_length = 10000


new_level = generate_bar_level(5000)
# new_level = generate_planar_bar_level(5000)
s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant))
worm_neural_net_integration = get_worm_neural_net_integration(s)
s.worm_neural_net_integration = worm_neural_net_integration


# import sys
# old_stdout = sys.stdout
# log_file = open("evolution.txt","w")
# sys.stdout = log_file

# run_evolution_on_worm()
# watch_best_worm()

run_pso_on_worm()
# watch_best_worm()

# while 1:
#
#
#
#     new_level = generate_bar_level(5000)
#     # new_level = generate_planar_bar_level(5000)
#     s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant))
#
#     # worm_neural_net_integration = get_worm_neural_net_integration(s)
#
#     graphics.user_control = True
#
#     s.run(graphics)