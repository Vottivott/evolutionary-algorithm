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
from radar_system import RadarSystem
from smoke import Smoke


class CopterSimulation:
    def __init__(self, level, copter, radar_system):
        self.level = level
        self.copter = copter
        self.smoke = None
        self.radar_system = radar_system
        self.gravity = np.array([[0.0],[0.2*9.8]])
        self.delta_t = 1.0/4
        self.space_pressed = False
        self.force_when_fire_is_on = np.array([[0.0],[0.2*-20]])
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
            if self.neural_net_integration is not None:
                self.neural_net_integration.run_network(self)
            elif graphics and user_control:
                self.copter.firing = self.space_pressed and not self.copter.exploded
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
    graphics = Graphics()
    view_offset = 1200/7.0
    level = generate_level(10000)
    s = CopterSimulation(level, Copter(np.array([[view_offset], [level.y_center(view_offset)]]), 20), RadarSystem())
    neural_net_integration = evocopter_neural_net_integration(s)
    s.set_neural_net_integration(neural_net_integration)

    class CopterFitnessFunction:
        def __init__(self):
            self.last_generation = -1
        def evaluate(self, variables, generation):
            if generation != self.last_generation:
                last_generation = generation
                s.level = generate_level(10000) # generate a new level for every generation
            neural_net_integration.set_weights(variables)
            s.copter = Copter(np.array([[view_offset], [level.y_center(view_offset)]]), 20) # new copter
            return s.run()

    vars = neural_net_integration.get_number_of_variables()
    var_size = 30
    m = vars * var_size

    ga = GeneticAlgorithm(30,
                          CopterFitnessFunction(),
                          TournamentSelection(0.75, 3),
                          SinglePointCrossover(0.9),
                          BinaryMutation(7.0 / m),
                          Elitism(1),
                          BinaryDecoding(5, vars, var_size),
                          BinaryInitialization(m))


    def callback(p):
        # if p.generation == 100:
        print str(p.generation) + ": " + str(p.best_fitness)
        if p.generation % 10 == 0:
            neural_net_integration.set_weights(p.best_variables)
            s.copter = Copter(np.array([[view_offset], [level.y_center(view_offset)]]), 20)  # new copter
            return s.run(graphics)
        # print p.best_variables


    ga.run(100, callback)
    # print "Average fitness over 200 runs:" + str(average_fitness(ga, 100, 200))



    # while 1:
    #     s.copter = Copter(np.array([[view_offset], [level.y_center(view_offset)]]), 20)
    #     s.level = generate_level(10000)
    #
    #     print s.run()