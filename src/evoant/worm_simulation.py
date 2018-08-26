import numpy as np

from evo_stats_handler import EvoStatsHandler
from pso_stats_handler import PSOStatsHandler

from mail import send_mail_message_with_image, send_mail_message
from genetic.decoding.real_number import RealNumberDecoding
from genetic.initialization.real_number import RealNumberInitialization
from pso.algorithm import ParticleSwarmOptimizationAlgorithm, PSOPopulationData
from ..stats_data_io import save_stats, load_stats
from worm_radar_system import WormRadarSystem
from worm import Worm
from genetic.algorithm import GeneticAlgorithm, PopulationData
from genetic.crossover.single_point import SinglePointCrossover
from genetic.decoding.binary import BinaryDecoding
from genetic.elitism.elitism import Elitism
from genetic.initialization.binary import BinaryInitialization
from genetic.mutation.binary import BinaryMutation
from genetic.mutation.creep import CreepMutation
from genetic.selection.tournament import TournamentSelection
from worm_graphics import WormGraphics
from bar_level import generate_bar_level_with_stones, generate_planar_bar_level, get_soccer_level
from ..neural_net_integration import evocopter_neural_net_integration, black_neural_net_integration
from ..population_data_io import save_population_data, load_population_data, get_latest_generation_number
from ..temp_data_io import wait_and_open_temp_data, save_temp_fitness
# from radar_system import RadarSystem, EnemysRadarSystem
from ..score_colors import get_color_from_score
from ..shot import Shot
from ..smoke import Smoke


import pygame
import time

from worm_neural_net_integration import get_worm_neural_net_integration

import sys




enemy_mode = True
view_offset = 1200 / 7.0
enemy_view_offset = 6.0 * 1200 / 7.0

base_start_x = 120 # For corridor

# base_start_x = 1200
enemy_width = 20
start_x = base_start_x + view_offset
min_x = base_start_x+view_offset+5*enemy_width

ball_radius = 10.0
segment_size = 13.0#17.0
num_segments = 13#12#4#3 #6 # worm_b=6
ball_ball_restitution = 1.0#0.4
ball_ground_restitution = 0.7
ball_ground_friction = 0.0#0.4
ball_mass = 10.0
spring_constant = 30.0


class WaterModulator:
    def __init__(self):
        self.cycle_length = 40.0
        self.acc_amplitude = 0.2
        self.x_offset_range = 200.0

    def apply(self, ball, timestep, delta_time):
        offset = ball.position[0] / self.x_offset_range
        water_acc = self.acc_amplitude * np.sin(offset + timestep / self.cycle_length)
        ball.velocity[1] += water_acc * delta_time


class WormSimulation:
    def __init__(self, level, worm):
        self.level = level
        self.worm = worm
        self.gravity = np.array([[0.0],[0.6*9.8]])
        self.delta_t = 1.0/4
        self.graphics = None
        self.score = 0.0
        self.timestep = 0
        self.time_since_improvement = 0
        self.left_neural_net_integration = None
        self.right_neural_net_integration = None
        self.water_modulator = WaterModulator()

    def termination_condition(self):
        # return self.time_since_improvement > 600 #0
        # return self.timestep > 300
        # return (not (self.graphics and self.graphics.user_control)) and self.timestep > 150
        # return (not (self.graphics and self.graphics.user_control)) and self.timestep > 200
        if self.worm.football.position[0] >= self.level.right_goal_x:
            self.score = 500.0 * (
            self.worm.football.position[0] - (self.level.left_goal_x + self.level.game_width / 2.0)) / (
                         self.level.game_width / 2.0)
            self.score = 500.0 + 1000.0 - self.timestep
            # self.score = 1000.0 - self.timestep
            return True
        if self.worm.football.position[0] <= self.level.left_goal_x:
            self.score = -500.0 -(1000.0 - self.timestep)
            # self.score = -(1000.0 - self.timestep)
            return True
        # return (not (self.graphics and self.graphics.user_control)) and self.timestep > 1000
        if (not (self.graphics and self.graphics.user_control)) and self.timestep > 1000:
            self.score = 500.0 * (self.worm.football.position[0] - (self.level.left_goal_x + self.level.game_width / 2.0)) / (self.level.game_width / 2.0)

            # if self.score == 0.0:
            #     self.score = 10.0 * (self.worm.football.position[0] - (self.level.left_goal_x + self.level.game_width / 2.0)) / (self.level.game_width / 2.0)
            return True
        # return False

    def run(self, graphics=None):
        self.graphics = graphics
        self.timestep = 0
        self.score = 0.0
        self.time_since_improvement = 0

        while not self.termination_condition():

            if self.left_neural_net_integration is not None:
                self.left_neural_net_integration.run_network(self)
                if self.right_neural_net_integration is not None:
                    self.right_neural_net_integration.run_network(self)
            elif self.graphics is not None and self.graphics.user_control:
                if self.graphics.keys is not None:
                    acc = 1.0
                    right = (self.graphics.keys[pygame.K_RIGHT] - self.graphics.keys[pygame.K_LEFT]) * acc
                    down = (self.graphics.keys[pygame.K_DOWN] - self.graphics.keys[pygame.K_UP]) * acc
                    connect = self.graphics.keys[pygame.K_SPACE]
                    # self.worm.fish[0].reaching = connect and 1.0 or 0.0
                    for i in range(len(self.worm.fish)):
                        # self.worm.fish[i].velocity[0] += right
                        # self.worm.fish[i].velocity[1] += down
                        self.worm.fish[i].reaching = connect and 1.0 or 0.0

                    self.worm.fish[0].velocity[0] += right
                    self.worm.fish[0].velocity[1] += down
                    # key_names = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9, pygame.K_0]
                    # keys = [self.graphics.keys[k] for k in key_names]
                    # for i,m in enumerate(self.worm.muscles):
                    #     m.target_length = keys[i] and self.worm.muscle_flex_length or self.worm.muscle_extend_length
                    # for i, b in enumerate(self.worm.balls):
                    #     b.grippingness = keys[self.worm.num_balls-1 + i] and 1.0 or 0.0


            self.worm.step(self.level, self.gravity, self.delta_t)

            # if all(f.position[0] >= self.level.corridor_end_x for f in self.worm.fish):
            #     print self.timestep
            #     return

            # TEST Water-like
            # for b in self.worm.fish:
            #     self.water_modulator.apply(b, self.timestep, self.delta_t)


            if self.graphics:
                space, enter, ctrl = self.graphics.update(self)
                # self.worm.muscles[0].target_length = space and self.worm.muscle_flex_length or self.worm.muscle_extend_length
                # self.worm.balls[-1].grippingness = ctrl
                # if not ctrl:
                #     self.worm.balls[-1].gripping = False
                if enter:
                    return

            # potential_score = self.worm.get_distance_travelled()
            # if potential_score > self.score:
            #     self.time_since_improvement = 0
            #     self.score = np.copy(potential_score)
            # else:
            #     self.time_since_improvement += 1
            # for f in self.worm.fish:
            #     if not f.has_scored and f.position[0] >= self.level.corridor_end_x:
            #         self.score += 500.0 - self.timestep
            #         f.has_scored = True

            self.timestep += 1




def get_corridor_fish_start_pos(lvl, i):
    return np.array(lvl.combined_initial_pos[i])


def run_evaluation(level, fitness_calculator, use_graphics=False):
    s.level = level
    s.left_neural_net_integration = left_neural_net_integration
    if s.left_neural_net_integration is not None:
        s.left_neural_net_integration.initialize_h()
    s.worm = Worm([get_corridor_fish_start_pos(level, i) for i in range(num_segments+1)], ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant, level.football_initial_position, level.football_initial_y_velocity)
    # s.worm = Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant)
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
    left_neural_net_integration.set_weights_and_possibly_initial_h(variables)
    if right_neural_net_integration is not None:
        right_neural_net_integration.set_weights_and_possibly_initial_h(enemy_variables)
    def fitness_calculator(sim):
        return sim.score
    return run_evaluations(levels, fitness_calculator, use_graphics)



def generate_levels(close_end=True):
    result = []
    # np.random.seed(0)
    for i in range(num_levels):
        # level = generate_bar_level_with_stones(level_length, close_end)
        # level = generate_planar_bar_level(level_length, close_end)
        level = get_soccer_level(1200, num_segments+1, True)
        result.append(level)
    return result


class WormFitnessFunction:
    def __init__(self):
        self.last_generation = -1
        self.debug_ind_n = 1

    def evaluate(self, variables, generation):

        #ONLY FOR EVO140 Football Second Neural Net
        if generation == 0:
            num_levels = 5
            print "num_levels = %d" % num_levels
        if generation == 10:
            num_levels = 10
            print "num_levels = %d" % num_levels
        if generation == 20:
            num_levels = 15
            print "num_levels = %d" % num_levels
        if generation == 30:
            num_levels = 20
            print "num_levels = %d" % num_levels
        if generation == 50:
            num_levels = 25
            print "num_levels = %d" % num_levels
        if generation == 80:
            num_levels = 30
            print "num_levels = %d" % num_levels



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



def run_evolution_on_worm(multiprocess_num_processes=1, multiprocess_index=None):

    fitness_function = WormFitnessFunction()

    if multiprocess_num_processes > 1:
        print "PROCESS " + str(multiprocess_index) + " OUT OF " + str(range(multiprocess_num_processes))
        if multiprocess_index > 0:
            fitness_process(multiprocess_num_processes, multiprocess_index, fitness_function)
            return

    # enemy_population_data = load_population_data(enemy_subfoldername, -1)
    # enemy_neural_net_integration.set_weights_and_possibly_initial_h(enemy_population_data.best_variables)
    # load_latest_enemy_network()

    vars = left_neural_net_integration.get_number_of_variables()


    # var_size = 30
    # m = vars * var_size

    population_size = 140#80
    mutate_c = 2.0#1.5
    crossover_p = 0.75

    send_mail_message(worm_subfoldername, special_message + "\n\n" + "population_size = " + str(population_size) + "\nmutate_c = " + str(mutate_c) + "\ncrossover_p = " + str(crossover_p))

    ga = GeneticAlgorithm(population_size,#150,
                          fitness_function,
                          TournamentSelection(0.75, 3),
                          SinglePointCrossover(crossover_p),#0.9),
                          CreepMutation(mutate_c / vars, 0.8, 0.005, True), #BinaryMutation(2.0 / m),
                          Elitism(1),
                          RealNumberDecoding(5.0), #BinaryDecoding(5, vars, var_size),
                          RealNumberInitialization(vars))# BinaryInitialization(m))

    def worm_callback(p, watch_only=False):

        # # if p.generation == 100:
        if not watch_only:
            #if p.generation % 50 == 0: # Save only every 50th generation
            save_population_data(worm_subfoldername, p, keep_last_n=10, keep_mod = 10)
            save_stats(worm_subfoldername, stats_handler, p)
            stats = load_stats(worm_subfoldername)
            if stats is not None:# and (
                    # len(stats["best_fitness_all_time"]) < 2 or stats["best_fitness_all_time"][-1] != stats["best_fitness_all_time"][-2]):
                stats_handler.produce_graph(stats, worm_subfoldername + ".png")
                msg = str(float(p.best_fitness)) + "\ngeneration " + str(p.generation)
                send_mail_message_with_image(worm_subfoldername, msg, worm_subfoldername + ".png", image_title="Gen: " + str(int(p.generation)) + "  Score: " + str(int(p.best_fitness)))
        average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
        print "\n[ " + str(p.generation) + ": " + str(
            p.best_fitness) + " : " + str(
            average_fitness) + " ]\n"

        # if p.generation % 10 == 0:
        #     global enemy_variables
        #     enemy_variables = load_population_data(enemy_subfoldername, p.generation).best_variables
        #     print "Enemy team updated to team " + str(p.generation)


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
            ga.run(None, worm_callback, population_data=worm_population_data, multiprocess_num_processes=multiprocess_num_processes, multiprocess_index=multiprocess_index, subfolder_name=worm_subfoldername)

def run_pso_on_worm(load_population_name="global", load_population_generation=-1):
    if load_population_name == "global":
        load_population_name = worm_subfoldername

    num_vars = left_neural_net_integration.get_number_of_variables()

    swarm_size = 35
    x_min = -1.5
    x_max = 1.5
    v_max = 8.0
    alpha = 0.3
    delta_t = 0.5
    initial_inertia_weight = 1.4

    send_mail_message(worm_subfoldername, "swarm_size = " + str(swarm_size) + "\nx_min = " + str(x_min) + "\nx_max = " + str(x_max) + "\nv_max = " + str(v_max) + "\nalpha = " + str(alpha) + "\ndelta_t = " + str(delta_t) + "\ninitial_inertia_weight = " + str(initial_inertia_weight))

    pso = ParticleSwarmOptimizationAlgorithm(swarm_size,#30,  # swarm_size
                                             num_vars,  # num_variables
                                             WormFitnessFunction(),  # fitness_function
                                             x_min, x_max,  # x_min, x_max
                                             v_max,#8.0,  # v_max
                                             alpha,  # alpha
                                             delta_t,  # delta_t
                                             2.0,  # cognition
                                             2.0,  # sociability
                                             initial_inertia_weight,  # initial_inertia_weight
                                             0.995,  # inertia_weight_decay
                                             0.35)  # min_inertia_weight

    def worm_callback(p, watch_only=False):

        # # if p.generation == 100:
        if not watch_only:
            #if p.generation % 50 == 0: # Save only every 50th generation
            save_population_data(worm_subfoldername, p, keep_last_n=4, keep_mod = None)
            save_stats(worm_subfoldername, stats_handler, p)
            stats = load_stats(worm_subfoldername)
            if stats is not None and (len(stats["best_fitness"]) < 2 or stats["best_fitness"][-1] != stats["best_fitness"][-2]):
                stats_handler.produce_graph(stats, worm_subfoldername + ".png")
                msg = str(float(p.best_fitness)) + "\ngeneration " + str(p.generation)
                send_mail_message_with_image(worm_subfoldername, msg, worm_subfoldername + ".png", image_title="Gen: " + str(int(p.generation)) + "  Score: " + str(int(p.best_fitness)))

        average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
        print "\n[ " + str(p.generation) + ": " + str(
            p.best_fitness) + " : " + str(
            average_fitness) + " ]     inertia = " + str(p.inertia_weight) + "\n"
        # if watch_only or (graphics is not None):# and p.generation % 10 == 0):
        #     fitness = run_worm_evaluation(p.best_variables, True)
        #     print "Fitness: " + str(fitness)

    watch_only = False
    global worm_population_data
    worm_population_data = load_population_data(load_population_name, load_population_generation)

    if isinstance(worm_population_data, PopulationData): # If the population data is of the wrong type
        # Convert from evolution population to PSO swarm by choosing the n best chromosomes, where n is the swarm size
        p = worm_population_data
        velocities = [alpha/delta_t * (-(x_max - x_min)/2.0 + np.random.random((num_vars, 1)) * (x_max - x_min)) for _ in range(swarm_size)]
        sorted_descending_indices = sorted(range(swarm_size), key=lambda i: p.fitness_scores[i], reverse=True)
        sorted_clipped_indices = sorted_descending_indices[:swarm_size]
        positions = [p.decoded_variable_vectors[i] for i in sorted_clipped_indices]
        fitness_scores = [p.fitness_scores[i] for i in sorted_clipped_indices]
        worm_population_data = PSOPopulationData(p.generation, positions, velocities, fitness_scores, [None for _ in range(swarm_size)],
                                                 [float('-inf') for _ in range(swarm_size)], None, float('-inf'), initial_inertia_weight)

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
    # worm_population_data = load_population_data(worm_subfoldername+"/pruned", 3)
    global levels
    levels = generate_levels()
    fitness = run_worm_evaluation(worm_population_data.best_variables, True)
    print "Fitness: " + str(fitness)

def load_latest_worm_network():
    worm_population_data = load_population_data(worm_subfoldername, -1)
    global left_neural_net_integration
    left_neural_net_integration.set_weights_and_possibly_initial_h(worm_population_data.best_variables)


# def multiprocess_evaluations(self, subfolder_name, decoded_variable_vectors, generation, num_processes,
#                              multiprocess_index):


def fitness_process(multiprocess_num_processes, multiprocess_index, fitness_function):
    def extract_function(function_or_object_with_function, function_name):
        x = function_or_object_with_function
        return getattr(x, function_name, x)
    evaluate = extract_function(fitness_function, "evaluate")

    global worm_population_data
    worm_population_data = load_population_data(worm_subfoldername, -1)
    if worm_population_data is None:
        generation = 0
        population_size = 140 # TODO: Remove this temporary code
        worm_population_data = None
    else:
        generation = worm_population_data.generation
        population_size = len(worm_population_data.population)
        worm_population_data = None # free memory

    while 1:

        next_generation, decoded_variable_vectors = None, None
        print "Process " + str(multiprocess_index) + " waiting for generation_and_decoded_variable_vectors for generation " + str(generation+1) + "..."
        t0 = time.time()
        while 1: # Wait until correct generation_and_decoded_variable_vectors file is in temp folder
            next_generation, decoded_variable_vectors = wait_and_open_temp_data(worm_subfoldername, "generation_and_decoded_variable_vectors")
            if next_generation == generation + 1:
                break
            time.sleep(3)
        t = time.time() - t0
        print "Process " + str(multiprocess_index) + " waited for " + str(
            t) + " seconds."
        print "Process " + str(
            multiprocess_index) + " starting fitness evaluations..."
        t0 = time.time()
        for individual_index in range(multiprocess_index, population_size, multiprocess_num_processes):
            individual_fitness = evaluate(decoded_variable_vectors[individual_index], next_generation)
            save_temp_fitness(worm_subfoldername, individual_index, individual_fitness)
        t = time.time() - t0
        print "Process " + str(multiprocess_index) + " evaluated its individuals in " + str(
            t) + " seconds."

        generation += 1



levels = []


level_length = 10000

new_level = get_soccer_level(1200, num_segments+1, True)
# new_level = get_soccer_level(1200, num_segments+1, True)



# new_level = generate_bar_level_with_stones(5000)
# new_level = generate_planar_bar_level(5000)
# s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant))
s = WormSimulation(new_level,
                   Worm([get_corridor_fish_start_pos(new_level, i) for i in range(num_segments + 1)], ball_radius,
                        segment_size, num_segments, ball_ball_restitution, ball_ground_restitution,
                        ball_ground_friction, ball_mass, spring_constant, new_level.football_initial_position, new_level.football_initial_y_velocity))

left_neural_net_integration = get_worm_neural_net_integration(s, version=2) # Second neural net version
s.left_neural_net_integration = left_neural_net_integration

# right_neural_net_integration = None
right_neural_net_integration = get_worm_neural_net_integration(s, mirrored = True, version=1)
s.right_neural_net_integration = right_neural_net_integration


# import sys
# old_stdout = sys.stdout
# log_file = open("evolution.txt","w")
# sys.stdout = log_file


# worm_subfoldername = "worm_b"
# worm_subfoldername = "worm_2segs_planar"
# worm_subfoldername = "PSO_worm_3segs_planar"
# worm_subfoldername = "PSO_worm_6segs_planar"
# worm_subfoldername = "PSO_worm_1seg"
# worm_subfoldername = "EVO150 Doorway"
# worm_subfoldername = "PSO35 Large Doorway"
# worm_subfoldername = "EVO80 Large Doorway"
# worm_subfoldername = "PSO35 Football from EVO80 41"
# worm_subfoldername = "EVO80 Football 1" # Against static enemy, with random ball velocity ; 42 num_levels=5, against 41
worm_subfoldername = "EVO140 Football Second Neural Net" # Against team 188 from "EVO80 Football 1", mutate_c=2, num_levels=15
print worm_subfoldername

special_message = ""

num_levels = 15#5 #REMEMBER TO SET CORRECTLY   #10#30  #14#7#5#1#4#30#15#4#30#15



enemy_subfoldername = "EVO80 Football 1"
# g = ((get_latest_generation_number(enemy_subfoldername)) / 10)*10
g = 113   #188#174 # som hade 720
enemy_variables = load_population_data(enemy_subfoldername, g).best_variables
print "Enemy team set to team " + str(g)

# from evomath import *
# for i in range(44, 58+1):
#     p = load_population_data(worm_subfoldername, i)
#     print p.generation, p.best_fitness, p.fitness_scores[:3], len(p.fitness_scores)
# exit()

stats_handler = EvoStatsHandler(); run_evolution_on_worm(multiprocess_num_processes=7, multiprocess_index=0)
#stats_handler = EvoStatsHandler(); run_evolution_on_worm(multiprocess_num_processes=3, multiprocess_index=2)
# stats_handler = PSOStatsHandler(); run_pso_on_worm()#"EVO80 Football 1", 41)

graphics = WormGraphics(); graphics.who_to_follow = None
# graphics = None
while 1:
    watch_best_worm()
# exit(0)



while 1:



    # new_level = generate_bar_level_with_stones(5000)
    # new_level = generate_planar_bar_level(5000)
    # np.random.seed(0)
    new_level = get_soccer_level(1200, num_segments+1, True)
    graphics.who_to_follow = None

    s = WormSimulation(new_level, Worm([get_corridor_fish_start_pos(new_level, i) for i in range(num_segments+1)], ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant, new_level.football_initial_position, new_level.football_initial_y_velocity))
    # s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, ball_mass, spring_constant))

    # worm_neural_net_integration = get_worm_neural_net_integration(s)

    graphics.user_control = True

    s.run(graphics)