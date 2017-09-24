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
from neural_net_integration import evocopter_neural_net_integration
from population_data_io import save_population_data, load_population_data
from radar_system import RadarSystem, EnemysRadarSystem
from shot import Shot
from smoke import Smoke


MAIN = 'main'

class EnemyInstance:
    def __init__(self, enemy, enemy_neural_net_integration):
        self.enemy = enemy  # [Enemy(np.array([[800], [level.y_center(800)]]), 20), Enemy(np.array([[890], [level.y_center(890)]]), 20)]
        if enemy_neural_net_integration is None:
            self.h = None
        else:
            self.h = enemy_neural_net_integration.get_empty_h()
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
        self.enemy_passed_dist = 300
        self.enemy_intro_dist = 1030

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

    def set_enemy_neural_net_integration(self, neural_net_integration):
        self.enemy_neural_net_integration = neural_net_integration

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

    def run(self, graphics=None, user_control=None):
        self.graphics = graphics
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
                if user_control == MAIN:
                    if self.user_control_main(graphics):
                        return True
                else:
                    if self.user_control_enemy(graphics, user_control):
                        return True
            elif user_control is None and graphics and self.copter.exploded and abs(self.copter.velocity[0]) < 0.1:
                return self.copter.get_x()
            if user_control != MAIN and self.main_neural_net_integration is not None and not self.copter.exploded:
                self.main_neural_net_integration.run_network(self)
            for enemy_index in range(len(self.enemy_instances)):
                if user_control != enemy_index and self.enemy_neural_net_integration is not None and not self.enemy_instances[enemy_index].enemy.exploded:
                    self.enemy_neural_net_integration.run_network(self, enemy_index)



            if self.copter.firing:
                fire_force = self.force_when_fire_is_on
            else:
                fire_force = self.force_when_fire_is_off
            still_flying = self.copter.step(self.level, self.gravity, fire_force, self.delta_t)

            self.add_newcoming_enemies()
            self.remove_passed_enemy_instances()

            enemy_still_flying = [True] * len(self.enemy_instances)

            if not self.copter.exploded:
                for ei in self.enemy_instances:
                    if not ei.enemy.exploded and self.copter.collides_with(ei.enemy):
                        still_flying = False

            for i,ei in enumerate(self.enemy_instances):
                if ei.enemy.firing:
                    enemy_fire_force = self.force_when_fire_is_on
                else:
                    enemy_fire_force = self.force_when_fire_is_off
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

            self.timestep += 1
            if graphics:

                if not still_flying:
                    if not self.copter.exploded:
                        self.smoke.create_explosion()
                        graphics.play_crash_sound()
                        self.copter.exploded = True
                        start_x = base_start_x + view_offset
                        print "Fitness: " + str(self.copter.position[0] - start_x)
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

            elif not still_flying:
                return self.copter.get_x()  # Return the distance travelled = the fitness score


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

def get_enemy_instance(position, graphics):
    ei = EnemyInstance(
        Enemy(np.array([position[0], position[1]])), None)
    if graphics:
        ei.smoke = Smoke(ei.enemy.position, 4, 0.05, s.gravity, graphics.enemy_smoke_color)
    return ei




if __name__ == "__main__":
    level_length = 300000
    graphics = Graphics()
    view_offset = 1200 / 7.0
    enemy_view_offset = 6.0 * 1200 / 7.0
    new_level = generate_level(level_length)
    s = CopterSimulation(new_level, Copter(np.array([[view_offset], [new_level.y_center(view_offset)]]), 20), RadarSystem())
    neural_net_integration = evocopter_neural_net_integration(s)
    s.set_main_neural_net_integration(neural_net_integration)

    class CopterFitnessFunction:
        def __init__(self):
            self.last_generation = -1
        def evaluate(self, variables, generation):
            if generation != self.last_generation:
                last_generation = generation
                s.level = generate_level(level_length) # generate a new level for every generation
            neural_net_integration.set_weights(variables)
            start_x = base_start_x + view_offset
            s.copter = Copter(np.array([[start_x], [s.level.y_center(start_x)]]), 20) # new copter
            return s.run() - start_x # use moved distance from start point as fitness score

    vars = neural_net_integration.get_number_of_variables()
    var_size = 30
    m = vars * var_size

    ga = GeneticAlgorithm(80,
                          CopterFitnessFunction(),
                          TournamentSelection(0.75, 3),
                          SinglePointCrossover(0.9),
                          BinaryMutation(7.0 / m),
                          Elitism(2),
                          BinaryDecoding(5, vars, var_size),
                          BinaryInitialization(m))




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

    def callback(p):
        # if p.generation == 100:
        average_fitness = sum(p.fitness_scores) / len(p.fitness_scores)
        print str(p.generation) + ": " + str(p.best_fitness) + "       average fitness in population: " + str(average_fitness)
        save_population_data(subfoldername, p, keep_last_n=10)

        # f.append(p.best_fitness)
        # avg_f.append(average_fitness)
        # x.append(len(f))
        # graph_f.set_ydata(f)
        # graph_f.set_xdata(x)
        # graph_avg_f.set_ydata(avg_f)
        # graph_avg_f.set_xdata(x)
        # ax.set_xlim((0, len(f)+1))
        # fig.canvas.draw()

        if graphics is not None and p.generation % 10 == 0:
            neural_net_integration.set_weights(p.best_variables)
            s.copter = Copter(np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)  # new copter
            return s.run(graphics)
        # print p.best_variables


    player = MAIN

    user_play = None#player
    run_loaded_chromosome = True

    graphics.who_to_follow = MAIN#player

    enemy_mode = True

    base_start_x = 1200
    short_level_length = base_start_x + 5000
    num_enemies = 5
    num_short_levels = 7
    enemy_width = 20


    if enemy_mode:
        if user_play != None:
            while 1:
                s.level = generate_level(level_length)
                s.copter = Copter(
                    np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)

                # s.enemy_instances = []
                # s.enemy_instances.append(EnemyInstance(Enemy(np.array([[base_start_x + enemy_view_offset], [s.level.y_center(base_start_x + enemy_view_offset)]])),\
                #                          None))

                # s.enemies[0] = Enemy(np.array([[base_start_x + enemy_view_offset], [s.level.y_center(base_start_x + enemy_view_offset)]]), 20)
                # s.enemies[1] = Enemy(np.array([[base_start_x+80 + enemy_view_offset], [s.level.y_center(base_start_x+80 + enemy_view_offset)]]), 20)

                population_data = load_population_data(subfoldername, -1)
                neural_net_integration.set_weights(population_data.best_variables)

                s.run(graphics, user_control=user_play)
        else:
            if run_loaded_chromosome:
                while 1:
                    s.level = generate_level(short_level_length)
                    s.copter = Copter(
                        np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)
                    s.enemy_instances = []
                    ep = get_enemy_positions(short_level_length, num_enemies, s.level, enemy_width, base_start_x)
                    s.enemy_instance_queue = [get_enemy_instance(pos, graphics) for pos in ep]
                    population_data = load_population_data(subfoldername, -1)
                    neural_net_integration.set_weights(population_data.best_variables)
                    s.run(graphics)
            else:
                ga.run(None, callback, population_data=load_population_data(subfoldername, -1))
    else:
        if user_play != None:
            while 1:
                s.level = generate_level(level_length)
                s.copter = Copter(
                    np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)

                # s.enemy_instances = []
                # s.enemy_instances.append(EnemyInstance(Enemy(np.array([[base_start_x + enemy_view_offset], [s.level.y_center(base_start_x + enemy_view_offset)]])),\
                #                          None))

                # s.enemies[0] = Enemy(np.array([[base_start_x + enemy_view_offset], [s.level.y_center(base_start_x + enemy_view_offset)]]), 20)
                # s.enemies[1] = Enemy(np.array([[base_start_x+80 + enemy_view_offset], [s.level.y_center(base_start_x+80 + enemy_view_offset)]]), 20)

                population_data = load_population_data(subfoldername, -1)
                neural_net_integration.set_weights(population_data.best_variables)

                s.run(graphics, user_control=user_play)
        else:
            if run_loaded_chromosome:
                while 1:
                    s.level = generate_level(level_length)
                    s.copter = Copter(
                        np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)
                    population_data = load_population_data(subfoldername, -1)
                    neural_net_integration.set_weights(population_data.best_variables)
                    s.run(graphics)
            else:
                ga.run(None, callback, population_data=load_population_data(subfoldername, -1))


