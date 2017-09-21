import numpy as np

from copter import Copter
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
from radar_system import RadarSystem
from smoke import Smoke






class CopterSimulation:
    def __init__(self, level, copter, radar_system):
        self.level = level
        self.copter = copter
        self.smoke = None
        self.radar_system = radar_system
        self.gravity = np.array([[0.0],[0.4*9.8]])
        self.delta_t = 1.0/4
        self.space_pressed = False
        self.force_when_fire_is_on = np.array([[0.0],[0.4*-20]])
        self.force_when_fire_is_off = np.array([[0.0],[0.0]])
        self.time_since_last_sputter_sound = 0
        self.sputter_sound_interval = 10
        self.timestep = 0
        self.neural_net_integration = None

    def set_neural_net_integration(self, neural_net_integration):
        self.neural_net_integration = neural_net_integration


    def run(self, graphics=None, user_control=False):
        if graphics:
            self.smoke = Smoke(self.copter.position, 4, 0.05, self.gravity)
        while 1:
            if user_control and graphics:
                self.copter.firing = self.space_pressed and not self.copter.exploded
            elif self.neural_net_integration is not None and not self.copter.exploded:
                self.neural_net_integration.run_network(self)
            else:
                self.copter.firing = False
            if self.copter.velocity[0] < 0.1:
                return True
            if self.copter.firing:
                fire_force = self.force_when_fire_is_on
            else:
                fire_force = self.force_when_fire_is_off
            still_flying = self.copter.step(self.level, self.gravity, fire_force, self.delta_t)
            self.timestep += 1
            if graphics:
                if not still_flying:
                    if not self.copter.exploded:
                        self.smoke.create_explosion()
                        graphics.play_crash_sound()
                        self.copter.exploded = True
                    firing = False
                sputter = self.smoke.step(self.level, self.delta_t, self.copter.firing)
                if sputter:
                    if self.time_since_last_sputter_sound >= self.sputter_sound_interval:
                        graphics.play_sputter_sound()
                        self.time_since_last_sputter_sound = 0
                        self.sputter_sound_interval = 3 + np.random.random()*2
                self.time_since_last_sputter_sound += 1
                self.space_pressed = graphics.update(self)
            elif not still_flying:
                return self.copter.get_x()  # Return the distance travelled = the fitness score

if __name__ == "__main__":
    level_length = 100000
    graphics = Graphics()
    view_offset = 1200/7.0
    level = generate_level(level_length)
    s = CopterSimulation(level, Copter(np.array([[view_offset], [level.y_center(view_offset)]]), 20), RadarSystem())
    neural_net_integration = evocopter_neural_net_integration(s)
    s.set_neural_net_integration(neural_net_integration)

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




    subfoldername = "7 counters (length 4), 15 radars (max_steps = 250, step_size = 4), velocity up+down"


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

        if p.generation % 10 == 0:
            neural_net_integration.set_weights(p.best_variables)
            s.copter = Copter(np.array([[view_offset], [s.level.y_center(view_offset)]]), 20)  # new copter
            return s.run(graphics)
        # print p.best_variables


    user_play = False
    run_loaded_chromosome = True

    if user_play:
        while 1:
            s.level = generate_level(level_length)
            s.copter = Copter(np.array([[view_offset], [s.level.y_center(view_offset)]]), 20)
            s.run(graphics, user_control=True)
    else:
        if run_loaded_chromosome:
            while 1:
                s.level = generate_level(level_length)
                s.copter = Copter(np.array([[view_offset], [s.level.y_center(view_offset)]]), 20)
                population_data = load_population_data(subfoldername, 118)
                neural_net_integration.set_weights(population_data.best_variables)
                s.run(graphics)
        else:
            ga.run(None, callback, population_data=load_population_data(subfoldername, 120))



