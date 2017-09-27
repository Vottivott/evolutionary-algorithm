import numpy as np

from copter import Copter
from enemy import Enemy
from genetic.algorithm import GeneticAlgorithm
from genetic.crossover.single_point import SinglePointCrossover
from genetic.decoding.binary import BinaryDecoding
from genetic.elitism.elitism import Elitism
from genetic.initialization.binary import BinaryInitialization
from genetic.mutation.binary import BinaryMutation
from genetic.selection.tournament import TournamentSelection
from graphics import Graphics
from level import generate_level
from neural_net_integration import evocopter_neural_net_integration, black_neural_net_integration
from population_data_io import save_population_data, load_population_data
from radar_system import RadarSystem, EnemysRadarSystem
from score_colors import get_color_from_score
from shot import Shot
from smoke import Smoke


MAIN = 'main'

class EnemyInstance:
    def __init__(self, enemy, en_neural_net_integration):
        self.enemy = enemy  # [Enemy(np.array([[800], [level.y_center(800)]]), 20), Enemy(np.array([[890], [level.y_center(890)]]), 20)]
        if en_neural_net_integration is None:
            self.h = None
        else:
            self.h = en_neural_net_integration.get_initial_h()
        self.smoke = None
        self.time_since_last_sputter_sound = 0
    def get_position(self):
        return self.enemy.position


class CopterSimulation:
    def __init__(self, level, copter, radar_system):
        self.level = level
        self.copter = copter
        self.smoke = None
        self.radar_system = radar_system
        self.enemys_radar_system = EnemysRadarSystem()
        self.shots = []
        self.gravity = np.array([[0.0],[0.4*9.8]])
        self.delta_t = 1.0/4
        self.space_pressed = False
        self.ctrl_pressed = False
        self.ctrl_on_press = False
        self.force_when_fire_is_on = np.array([[0.0], [0.4 * -20]])
        self.force_when_fire_is_off = np.array([[0.0],[0.0]])
        self.enemy_force_when_fire_is_on = np.array([[0.0], [0.4 * -20]])
        self.time_since_last_sputter_sound = 0
        self.sputter_sound_interval = 10
        self.timestep = 0
        self.main_neural_net_integration = None
        self.enemy_neural_net_integration = None
        self.enemy_passed_dist = 400
        self.enemy_intro_dist = 1050

        self.end_when_copter_dies = True
        self.end_when_enemy_dies = False
        self.end_when_all_enemies_die = False
        self.end_at_time = None

        self.number_of_enemy_deaths = 0


        self.total_enemy_living_time = 0.0

        self.enemy_instance_queue = [] # enemy instances not yet in view

        self.enemy_instances = []

        self.graphics = None

    def get_living_copter_list(self):
        if self.copter.exploded:
            return []
        else:
            return [self.copter]

    def get_living_enemy_instances(self, except_index=None):
        if except_index is not None:
            return [ei for i,ei in enumerate(self.enemy_instances) if not ei.enemy.exploded and not i==except_index]
        else:
            return [ei for ei in self.enemy_instances if not ei.enemy.exploded]

    def remove_passed_enemy_instances(self):
        for i,ei in enumerate(self.enemy_instances):
            if self.copter.position[0] - ei.get_position()[0] > self.enemy_passed_dist:
                del self.enemy_instances[i]

    def add_newcoming_enemies(self):
        i = 0
        while i < len(self.enemy_instance_queue):
            ei = self.enemy_instance_queue[i]
            pos = ei.get_position()
            if pos[0] - self.copter.position[0] < self.enemy_intro_dist:
                self.enemy_instances.append(ei)
                del self.enemy_instance_queue[i]
            else:
                i += 1

    def set_main_neural_net_integration(self, neural_net_integration):
        self.main_neural_net_integration = neural_net_integration

    def set_enemy_neural_net_integration(self, enemy_neural_net_integration):
        self.enemy_neural_net_integration = enemy_neural_net_integration

    def user_control_main(self, graphics):
        self.copter.firing = self.space_pressed and not self.copter.exploded
        if self.copter.exploded and abs(self.copter.velocity[0]) < 0.1:
            return True
        if self.ctrl_on_press:
            self.copter_shoot()
            self.ctrl_on_press = False

    def copter_shoot(self):
        self.copter.recoil()
        self.shots.append(Shot(self.copter.position, self.gravity))
        if self.graphics:
            self.smoke.create_shot_background()
            graphics.play_shot_sound()

    def user_control_enemy(self, graphics, index):
        enemy = self.enemy_instances[index].enemy
        enemy.firing = self.space_pressed and not enemy.exploded
        enemy.moving_left = self.left_pressed and not enemy.exploded
        if enemy.exploded and abs(enemy.velocity[0]) < 0.1:
            return True
        if self.ctrl_on_press:
            enemy.dive()
            graphics.play_enemy_dive_sound()
            self.ctrl_on_press = False

    def get_copter_distance_travelled(self):
        return self.copter.position[0] - base_start_x

    def mediate_smoke_particle_rate(self):
        particle_count = len(self.smoke.particles) + len(self.smoke.frozen_particles)*0.5 + sum(len(ei.smoke.particles) + len(ei.smoke.frozen_particles)*0.5 for ei in self.enemy_instances) or 1
        calculated_rate = min(70.0/particle_count, 3)
        for ei in self.enemy_instances:
            ei.smoke.particle_rate = calculated_rate
        self.smoke.particle_rate = min(2*calculated_rate, 4)


    def run(self, graphics=None, user_control=None):
        self.graphics = graphics
        self.timestep = 0
        self.number_of_enemy_deaths = 0
        self.total_enemy_living_time = 0.0
        if graphics:
            self.smoke = Smoke(self.copter.position, 4, 0.05, self.gravity, graphics.main_copter_smoke_color)
            # for enemy_instance in self.enemy_instances:
            #     enemy_instance.smoke = Smoke(enemy_instance.enemy.position, 4, 0.05, self.gravity, graphics.enemy_smoke_color)
        while 1:
            if graphics:
                ctrl_not_previously_pressed = not self.ctrl_pressed
                self.space_pressed, self.ctrl_pressed, self.left_pressed = graphics.update(self)
                if self.ctrl_pressed and ctrl_not_previously_pressed:
                    self.ctrl_on_press = True
            if user_control != None and graphics:
                if user_control == MAIN or user_control >= len(self.enemy_instances):
                    if self.user_control_main(graphics):
                        return True
                else:
                    if self.user_control_enemy(graphics, user_control):
                        return True
            elif user_control is None and graphics and self.copter.exploded and self.end_when_copter_dies and abs(self.copter.velocity[0]) < 0.1:
                # print "end at copter death!!"
                return self.get_copter_distance_travelled()
            if user_control != MAIN and self.main_neural_net_integration is not None and not self.copter.exploded:
                self.main_neural_net_integration.run_network(self)
            for enemy_index in range(len(self.enemy_instances)):
                if user_control != enemy_index:
                    if self.enemy_neural_net_integration is not None and not self.enemy_instances[enemy_index].enemy.exploded:
                        self.enemy_neural_net_integration.run_network(self, enemy_index, self.enemy_instances[enemy_index].h)
                    else:
                        if self.enemy_neural_net_integration is None:
                            print "no enemy_neural_net integration"
                        enemy  =self.enemy_instances[enemy_index].enemy
                        enemy.velocity = -0.25*self.gravity
                        enemy.firing = False

            self.total_enemy_living_time += len(self.get_living_enemy_instances())


            if self.copter.firing:
                fire_force = np.copy(self.force_when_fire_is_on)
            else:
                fire_force = np.copy(self.force_when_fire_is_off)
            still_flying = self.copter.step(self.level, self.gravity, fire_force, self.delta_t)

            self.add_newcoming_enemies()
            self.remove_passed_enemy_instances()

            enemy_still_flying = [True] * len(self.enemy_instances)

            for i, ei in enumerate(self.enemy_instances):
                if ei.enemy.firing:
                    enemy_fire_force = np.copy(self.enemy_force_when_fire_is_on)
                else:
                    enemy_fire_force = np.copy(self.force_when_fire_is_off)
                enemy_still_flying[i] = ei.enemy.step(self.level, self.gravity, enemy_fire_force, self.delta_t)

            for shot in self.shots:
                if not shot.step(self.level, self.delta_t):
                    self.shots.remove(shot)
                else:
                    hit = False
                    for i, ei in enumerate(self.enemy_instances):
                        if ei.enemy.collides_with(shot):
                            enemy_still_flying[i] = 'shot'
                            hit = True
                    if hit:
                        self.shots.remove(shot)

            for i,ei in enumerate(self.enemy_instances):
                if not ei.enemy.exploded:
                    if self.copter.collides_with(ei.enemy) and not self.copter.exploded:
                        still_flying = False
                        ei.enemy.velocity *= ei.enemy.collision_friction
                    for i_other in range(i+1, len(self.enemy_instances)):
                        ei_other = self.enemy_instances[i_other]
                        if not ei_other.enemy.exploded and ei.enemy.collides_with(ei_other.enemy):
                                enemy_still_flying[i] = False
                                enemy_still_flying[i_other] = False



            self.timestep += 1
            if self.end_at_time is not None and self.timestep >= self.end_at_time:
                # print "End at time!"
                return self.get_copter_distance_travelled()
            if graphics:
                self.mediate_smoke_particle_rate()
                if not still_flying:
                    if not self.copter.exploded:
                        self.smoke.create_explosion()
                        graphics.play_crash_sound()
                        self.number_of_enemy_deaths += 1
                        self.copter.exploded = True
                        start_x = base_start_x + view_offset
                        # print "Fitness: " + str(self.copter.position[0] - start_x)
                sputter = self.smoke.step(self.level, self.delta_t, self.copter.firing and not self.copter.exploded)
                if sputter:
                    if self.time_since_last_sputter_sound >= self.sputter_sound_interval:
                        graphics.play_sputter_sound()
                        self.time_since_last_sputter_sound = 0
                        self.sputter_sound_interval = 4
                self.time_since_last_sputter_sound += 1
                for i in range(len(self.enemy_instances)):
                    if not enemy_still_flying[i]:
                        if not self.enemy_instances[i].enemy.exploded:
                            self.enemy_instances[i].smoke.create_explosion()
                            graphics.play_crash_sound()
                            self.enemy_instances[i].enemy.exploded = True
                            #print "Fitness: " + str(self.copter.position[0])
                    elif enemy_still_flying[i] == 'shot' and not self.enemy_instances[i].enemy.exploded:
                        self.enemy_instances[i].smoke.create_explosion()
                        graphics.play_enemy_hit_sound()
                        self.enemy_instances[i].enemy.exploded = True
                    sputter = self.enemy_instances[i].smoke.step(self.level, self.delta_t, self.enemy_instances[i].enemy.firing)
                    if sputter:
                        if self.enemy_instances[i].time_since_last_sputter_sound >= self.sputter_sound_interval:
                            graphics.play_enemy_sputter_sound()
                            self.enemy_instances[i].time_since_last_sputter_sound = 0
                            self.sputter_sound_interval = 4
                    self.enemy_instances[i].time_since_last_sputter_sound += 1
                    if self.enemy_instances[i].enemy.diving:
                        graphics.play_enemy_dive_sound()
                        self.enemy_instances[i].enemy.diving = False
            else:
                if not still_flying:
                    if not self.copter.exploded:
                        self.number_of_enemy_deaths += 1
                        self.copter.exploded = True
                for i in range(len(self.enemy_instances)):
                    if not enemy_still_flying[i]:
                        if not self.enemy_instances[i].enemy.exploded:
                            self.enemy_instances[i].enemy.exploded = True
                    elif enemy_still_flying[i] == 'shot' and not self.enemy_instances[i].enemy.exploded:
                        self.enemy_instances[i].enemy.exploded = True

            if not graphics and not still_flying and self.end_when_copter_dies:
                # print "copter death end"
                return self.get_copter_distance_travelled()  # Return the distance travelled = the fitness score
            if self.end_when_enemy_dies:
                for ei in self.enemy_instances:
                    if ei.enemy.exploded:
                        return ei.enemy.position[0]
            if self.end_when_all_enemies_die:
                all_dead = True
                for ei in self.enemy_instances:
                    if not ei.enemy.exploded:
                        all_dead = False
                if all_dead:
                    return self.timestep




def too_close(x_positions, min_dist):
    for i in range(len(x_positions)):
        for j in range(i) + range(i+1,len(x_positions)):
            if abs(x_positions[i] - x_positions[j]) < min_dist:
                return True
    return False


def get_enemy_positions(short_level_length, num_enemies, level, enemy_width, min_x):
    x_positions = []
    while not x_positions or too_close(x_positions, enemy_width):
        x_positions = [min_x + np.random.random() * (short_level_length-min_x) for _ in range(num_enemies)]
    x_positions = sorted(x_positions)
    positions = [np.array([[x],[level.y_center(int(x))]]) for x in x_positions]
    return positions

def get_enemy_instance(position, graphics, neural_net_integration):
    ei = EnemyInstance(
        Enemy(np.array([position[0], position[1]])), neural_net_integration)
    if graphics:
        ei.smoke = Smoke(ei.enemy.position, 4, 0.05, s.gravity, graphics.enemy_smoke_color)
    return ei




graphics = None

enemy_mode = True
view_offset = 1200 / 7.0
enemy_view_offset = 6.0 * 1200 / 7.0
base_start_x = 1200
enemy_width = 20
start_x = base_start_x + view_offset
min_x = base_start_x+view_offset+5*enemy_width

copter_subfoldername = "copter"
enemy_subfoldername = "enemy"


short_level_length = base_start_x + 5000
num_enemies = 5
num_short_levels = 7

new_level = generate_level(short_level_length)
s = CopterSimulation(new_level, Copter(np.array([[start_x], [new_level.y_center(start_x)]]), 20),
                     RadarSystem())
neural_net_integration = evocopter_neural_net_integration(s)
s.set_main_neural_net_integration(neural_net_integration)

enemy_neural_net_integration = black_neural_net_integration(s)
s.set_enemy_neural_net_integration(enemy_neural_net_integration)


global short_levels_and_enemy_positions
short_levels_and_enemy_positions = [] # pairs of (level, enemy_positions) for each mini-level



s.end_at_time = 10000

def generate_mini_levels_and_enemy_positions():
    result = []
    for i in range(num_short_levels):
        level = generate_level(short_level_length)
        ep = get_enemy_positions(short_level_length, num_enemies, level, enemy_width, min_x)
        result.append((level, ep))
    return result

def run_evaluation(level, positions, fitness_calculator, use_graphics=False):
    s.level = level
    s.set_main_neural_net_integration(neural_net_integration)
    s.set_enemy_neural_net_integration(enemy_neural_net_integration)
    if s.main_neural_net_integration is not None:
        s.main_neural_net_integration.initialize_h()
    s.enemy_instances = []
    s.enemy_instance_queue = [
        get_enemy_instance(pos, graphics if use_graphics else None, s.enemy_neural_net_integration) for pos
        in positions]
    s.copter = Copter(np.array([[start_x], [s.level.y_center(start_x)]]), 20)  # new copter
    s.run(graphics if use_graphics else None)  # - start_x  # use moved distance from start point as fitness score
    # if watch_only:
    #     print fitness
    return fitness_calculator(s)

def run_evaluations(levels_and_enemy_positions, fitness_calculator, use_graphics=False):
    fitness_total = 0.0
    for level, positions in levels_and_enemy_positions:
        fitness_total += run_evaluation(level, positions, fitness_calculator, use_graphics)
    return fitness_total / num_short_levels


def watch_copter_vs_enemies():
    global graphics
    graphics = Graphics()
    while 1:
        global short_levels_and_enemy_positions
        short_levels_and_enemy_positions = generate_mini_levels_and_enemy_positions()


        copter_population_data = load_population_data(copter_subfoldername, -1)
        copter_variables = copter_population_data.best_variables

        load_latest_enemy_network()

        fitness = run_copter_evaluation(copter_variables, True)
        print "Average copter distance: " + str(fitness)

def run_enemy_evaluation(variables, use_graphics=False):
    enemy_neural_net_integration.set_weights_and_possibly_initial_h(variables)
    ENEMY_DEATH_PENALTY = 300.0
    def fitness_calculator(sim):
        return - sim.get_copter_distance_travelled() - ENEMY_DEATH_PENALTY * sim.number_of_enemy_deaths
    return run_evaluations(short_levels_and_enemy_positions, fitness_calculator, use_graphics)

def run_copter_evaluation(variables, use_graphics=False):
    neural_net_integration.set_weights_and_possibly_initial_h(variables)
    def fitness_calculator(sim):
        return sim.get_copter_distance_travelled()
    return run_evaluations(short_levels_and_enemy_positions, fitness_calculator, use_graphics)


def load_latest_enemy_network():
    enemy_population_data = load_population_data(enemy_subfoldername, -1)
    global enemy_neural_net_integration
    enemy_neural_net_integration.set_weights_and_possibly_initial_h(enemy_population_data.best_variables)

def load_latest_copter_network():
    copter_population_data = load_population_data(copter_subfoldername, -1)
    global neural_net_integration
    neural_net_integration.set_weights_and_possibly_initial_h(copter_population_data.best_variables)


class CopterFitnessFunction:
    def __init__(self):
        self.last_generation = -1

        self.debug_ind_n = 1

    def evaluate(self, variables, generation):
        if generation != self.last_generation:
            self.last_generation = generation
            global short_levels_and_enemy_positions
            short_levels_and_enemy_positions = generate_mini_levels_and_enemy_positions()
            load_latest_enemy_network()
            self.debug_ind_n = 1
        fitness = run_copter_evaluation(variables, False)
        print get_color_from_score(fitness, False) + str(int(fitness)),
        #print "("+str(self.debug_ind_n) + "): " + str(fitness)
        self.debug_ind_n += 1
        return fitness

class EnemyFitnessFunction:
    def __init__(self):
        self.last_generation = -1

        self.debug_ind_n = 1

    def evaluate(self, variables, generation):
        if generation != self.last_generation:
            self.last_generation = generation
            global short_levels_and_enemy_positions
            short_levels_and_enemy_positions = generate_mini_levels_and_enemy_positions()
            load_latest_copter_network()
            self.debug_ind_n = 1
        fitness = run_enemy_evaluation(variables, False)
        print get_color_from_score(fitness, True) + str(int(fitness)),
        #print "("+str(self.debug_ind_n) + "): " + str(fitness)
        self.debug_ind_n += 1
        return fitness

def run_evolution_on_enemy():
    # copter_population_data = load_population_data(copter_subfoldername, -1)
    # neural_net_integration.set_weights_and_possibly_initial_h(copter_population_data.best_variables)
    # load_latest_copter_network()
    s.end_when_copter_dies = True
    s.end_when_enemy_dies = False
    s.end_when_all_enemies_die = False

    vars = enemy_neural_net_integration.get_number_of_variables()
    var_size = 30
    m = vars * var_size

    ga = GeneticAlgorithm(80,
                          EnemyFitnessFunction(),
                          TournamentSelection(0.75, 3),
                          SinglePointCrossover(0.9),
                          BinaryMutation(7.0 / m),
                          Elitism(1),
                          BinaryDecoding(5, vars, var_size),
                          BinaryInitialization(m))

    def enemy_callback(p, watch_only=False):
        # if p.generation == 100:
        if not watch_only:
            save_population_data(enemy_subfoldername, p, keep_last_n=10)
        average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
        print get_color_from_score(p.best_fitness, True) + "\n[ " + str(p.generation) + ": " + str(
            p.best_fitness) + " : " + str(
            average_fitness) + " ]\n\n"
        # if watch_only or (graphics is not None and p.generation % 10 == 0):
        #     fitness = run_enemy_evaluation(p.best_variables, True)
        #     print "Fitness: " + str(fitness)


    watch_only = False
    enemy_population_data = load_population_data(enemy_subfoldername, -1)
    # g = enemy_population_data.best_individual_genes

    if True:
        if watch_only:
            while 1:
                # BinaryMutation(100.0 / m).mutate(g, 1)
                # enemy_population_data.best_variables = BinaryDecoding(5, vars, var_size).decode(g)
                global short_levels_and_enemy_positions
                short_levels_and_enemy_positions = generate_mini_levels_and_enemy_positions()
                enemy_callback(enemy_population_data, False)
                #watch_run(enemy_population_data)
        else:
            ga.run(None, enemy_callback, population_data=enemy_population_data)

def run_evolution_on_copter():

    # enemy_population_data = load_population_data(enemy_subfoldername, -1)
    # enemy_neural_net_integration.set_weights_and_possibly_initial_h(enemy_population_data.best_variables)
    # load_latest_enemy_network()
    s.end_when_copter_dies = True
    s.end_when_enemy_dies = False
    s.end_when_all_enemies_die = False

    vars = neural_net_integration.get_number_of_variables()
    var_size = 30
    m = vars * var_size

    ga = GeneticAlgorithm(80,
                          CopterFitnessFunction(),
                          TournamentSelection(0.75, 3),
                          SinglePointCrossover(0.9),
                          BinaryMutation(7.0 / m),
                          Elitism(1),
                          BinaryDecoding(5, vars, var_size),
                          BinaryInitialization(m))

    def copter_callback(p, watch_only=False):
        # if p.generation == 100:
        if not watch_only:
            save_population_data(copter_subfoldername, p, keep_last_n=10)
        average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
        print get_color_from_score(p.best_fitness, False) + "\n[ " + str(p.generation) + ": " + str(
            p.best_fitness) + " : " + str(
            average_fitness) + " ]\n"
        # if watch_only or (graphics is not None and p.generation % 10 == 0):
        #     fitness = run_copter_evaluation(p.best_variables, True)
        #     print "Fitness: " + str(fitness)

    watch_only = False
    copter_population_data = load_population_data(copter_subfoldername, -1)
    # g = copter_population_data.best_individual_genes

    if True:
        if watch_only:
            while 1:
                # BinaryMutation(100.0 / m).mutate(g, 1)
                # copter_population_data.best_variables = BinaryDecoding(5, vars, var_size).decode(g)
                global short_levels_and_enemy_positions
                short_levels_and_enemy_positions = generate_mini_levels_and_enemy_positions()
                copter_callback(copter_population_data, True)
                # watch_run(copter_population_data)
        else:
            ga.run(None, copter_callback, population_data=copter_population_data)




# subfoldername = "7 counters (length 4), 15 radars (max_steps = 250, step_size = 4), velocity up+down"
# subfoldername = "feedforward_larger_no_enemies"
subfoldername = "recurrent_no_enemies"


# Init plot

# import matplotlib.pyplot as plt
# x = []
# f = []
# avg_f = []
# fig, ax = plt.subplots()
# graphs1 = [graph_f] = ax.plot(x,f)
# graphs2 = [graph_avg_f] = ax.plot(x,avg_f)
# ax.set_ylim((0, 1))
# ax.set_xlim((0,len(f)+1))
# plt.ion()
# plt.show()

# def callback(p):
#     # if p.generation == 100:
#     average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
#     print str(p.generation) + ": " + str(p.best_fitness) + "       average fitness in population: " + str(average_fitness)
#     save_population_data(subfoldername, p, keep_last_n=10)
#
#     # f.append(p.best_fitness)
#     # avg_f.append(average_fitness)
#     # x.append(len(f))
#     # graph_f.set_ydata(f)
#     # graph_f.set_xdata(x)
#     # graph_avg_f.set_ydata(avg_f)
#     # graph_avg_f.set_xdata(x)
#     # ax.set_xlim((0, len(f)+1))
#     # fig.canvas.draw()
#
#     if graphics is not None and p.generation % 10 == 0:
#         neural_net_integration.set_weights_and_possibly_initial_h(p.best_variables)
#         s.copter = Copter(np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)  # new copter
#         return s.run(graphics)
#         # print p.best_variables


player = MAIN

user_play = None#player#player
run_loaded_chromosome = True

if graphics is not None:
    graphics.who_to_follow = player

# if False:#watch_only:
#
#
#     while 1:
#         s.enemy_neural_net_integration = black_neural_net_integration(s)#None
#         s.level = generate_level(short_level_length)
#         s.copter = Copter(
#             np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)
#         s.enemy_instances = []
#         ep = get_enemy_positions(short_level_length, 5, s.level, enemy_width,
#                                  base_start_x + view_offset + 5 * enemy_width)
#         s.enemy_instance_queue = [get_enemy_instance(pos, graphics, s.enemy_neural_net_integration) for pos in ep]
#
#         if user_play != MAIN:
#             # population_data = load_population_data(subfoldername, -1)
#             # neural_net_integration.set_weights_and_possibly_initial_h(population_data.best_variables)
#             pass
#         else:
#             s.set_main_neural_net_integration(None)
#
#         s.run(graphics, user_control=user_play)
#
