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

        self.enemy_h = []#[None] * 2
        self.enemy_smokes = []#[None] * 2
        self.enemies = []#[Enemy(np.array([[800], [level.y_center(800)]]), 20), Enemy(np.array([[890], [level.y_center(890)]]), 20)]
        self.time_since_last_enemy_sputter_sound = []#[0] * 2

        self.graphics = None


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
        enemy = self.enemies[index]
        enemy.firing = self.space_pressed and not enemy.exploded
        enemy.moving_left = self.left_pressed and not enemy.exploded
        if enemy.exploded and abs(enemy.velocity[0]) < 0.1:
            return True
        if self.ctrl_on_press:
            enemy.dive()
            #self.enemy_smokes[index].create_dive_background()
            graphics.play_enemy_dive_sound()
            self.ctrl_on_press = False

    def run(self, graphics=None, user_control=None):
        self.graphics = graphics
        if graphics:
            self.smoke = Smoke(self.copter.position, 4, 0.05, self.gravity, graphics.main_copter_smoke_color)
            for i,enemy in enumerate(self.enemies):
                self.enemy_smokes[i] = Smoke(enemy.position, 4, 0.05, self.gravity, graphics.enemy_smoke_color)
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
            for enemy_index in range(len(self.enemies)):
                if user_control != enemy_index and self.enemy_neural_net_integration is not None and not self.copter.exploded:
                    self.enemy_neural_net_integration.run_network(self, enemy_index)



            if self.copter.firing:
                fire_force = self.force_when_fire_is_on
            else:
                fire_force = self.force_when_fire_is_off
            still_flying = self.copter.step(self.level, self.gravity, fire_force, self.delta_t)


            enemy_still_flying = [True] * len(self.enemies)

            if not self.copter.exploded:
                for enemy in self.enemies:
                    if not enemy.exploded and self.copter.collides_with(enemy):
                        still_flying = False

            for i,enemy in enumerate(self.enemies):
                if enemy.firing:
                    enemy_fire_force = self.force_when_fire_is_on
                else:
                    enemy_fire_force = self.force_when_fire_is_off
                enemy_still_flying[i] = enemy.step(self.level, self.gravity, enemy_fire_force, self.delta_t)

            for shot in self.shots:
                if not shot.step(self.level, self.delta_t):
                    self.shots.remove(shot)
                else:
                    hit = False
                    for i, enemy in enumerate(self.enemies):
                        if enemy.collides_with(shot):
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
                        print "Fitness: " + str(self.copter.position[0])
                sputter = self.smoke.step(self.level, self.delta_t, self.copter.firing and not self.copter.exploded)
                if sputter:
                    if self.time_since_last_sputter_sound >= self.sputter_sound_interval:
                        graphics.play_sputter_sound()
                        self.time_since_last_sputter_sound = 0
                        self.sputter_sound_interval = 4
                self.time_since_last_sputter_sound += 1
                for i in range(len(self.enemy_smokes)):
                    if not enemy_still_flying[i]:
                        if not self.enemies[i].exploded:
                            self.enemy_smokes[i].create_explosion()
                            graphics.play_crash_sound()
                            self.enemies[i].exploded = True
                            #print "Fitness: " + str(self.copter.position[0])
                    elif enemy_still_flying[i] == 'shot' and not self.enemies[i].exploded:
                        self.enemy_smokes[i].create_explosion()
                        graphics.play_enemy_hit_sound()
                        self.enemies[i].exploded = True
                    sputter = self.enemy_smokes[i].step(self.level, self.delta_t, self.enemies[i].firing)
                    if sputter:
                        if self.time_since_last_enemy_sputter_sound[i] >= self.sputter_sound_interval:
                            graphics.play_enemy_sputter_sound()
                            self.time_since_last_enemy_sputter_sound[i] = 0
                            self.sputter_sound_interval = 4
                    self.time_since_last_enemy_sputter_sound[i] += 1

            elif not still_flying:
                return self.copter.get_x()  # Return the distance travelled = the fitness score

if __name__ == "__main__":
    level_length = 300000
    graphics = Graphics()
    view_offset = 1200 / 7.0
    enemy_view_offset = 6.0 * 1200 / 7.0
    level = generate_level(level_length)
    s = CopterSimulation(level, Copter(np.array([[view_offset], [level.y_center(view_offset)]]), 20), RadarSystem())
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
            s.copter = Copter(np.array([[view_offset], [level.y_center(view_offset)]]), 20) # new copter
            return s.run()

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
    subfoldername = "feedforward_larger_no_enemies"
    # subfoldername = "recurrent_no_enemies"


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
            s.copter = Copter(np.array([[view_offset], [s.level.y_center(view_offset)]]), 20)  # new copter
            return s.run(graphics)
        # print p.best_variables


    player = MAIN

    user_play = player
    run_loaded_chromosome = True

    graphics.who_to_follow = player

    base_start_x = 1200

    if user_play != None:
        while 1:
            s.level = generate_level(level_length)
            s.copter = Copter(np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)
            # s.enemies[0] = Enemy(np.array([[base_start_x + enemy_view_offset], [s.level.y_center(base_start_x + enemy_view_offset)]]), 20)
            # s.enemies[1] = Enemy(np.array([[base_start_x+80 + enemy_view_offset], [s.level.y_center(base_start_x+80 + enemy_view_offset)]]), 20)

            # population_data = load_population_data(subfoldername, 150)
            # neural_net_integration.set_weights(population_data.best_variables)

            s.run(graphics, user_control=user_play)
    else:
        if run_loaded_chromosome:
            while 1:
                s.level = generate_level(level_length)
                s.copter = Copter(np.array([[base_start_x + view_offset], [s.level.y_center(base_start_x + view_offset)]]), 20)
                population_data = load_population_data(subfoldername, -1)
                neural_net_integration.set_weights(population_data.best_variables)
                s.run(graphics)
        else:
            ga.run(None, callback, population_data=load_population_data(subfoldername, -1))



