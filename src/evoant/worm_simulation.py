import numpy as np

from worm_radar_system import WormRadarSystem
from worm import Worm
from enemy import Enemy
from genetic.algorithm import GeneticAlgorithm
from genetic.crossover.single_point import SinglePointCrossover
from genetic.decoding.binary import BinaryDecoding
from genetic.elitism.elitism import Elitism
from genetic.initialization.binary import BinaryInitialization
from genetic.mutation.binary import BinaryMutation
from genetic.selection.tournament import TournamentSelection
from worm_graphics import WormGraphics
from bar_level import generate_bar_level
from neural_net_integration import evocopter_neural_net_integration, black_neural_net_integration
from population_data_io import save_population_data, load_population_data
from radar_system import RadarSystem, EnemysRadarSystem
from score_colors import get_color_from_score
from shot import Shot
from smoke import Smoke

import sys


class WormSimulation:
    def __init__(self, level, worm):
        self.level = level
        self.worm = worm
        self.worm_radar_system = WormRadarSystem(worm.num_balls-1)
        self.gravity = np.array([[0.0],[0.4*9.8]])
        self.delta_t = 1.0/4
        self.graphics = None
        self.score = 0

    def run(self, graphics=None):
        self.graphics = graphics
        self.timestep = 0
        while 1:
            self.timestep += 1
            self.worm.step(self.level, self.gravity, self.delta_t)
            space, enter, ctrl = graphics.update(self)
            self.worm.muscles[0].target_length = space and 37.0 or 24.0
            self.worm.balls[-1].grippingness = ctrl
            if not ctrl:
                self.worm.balls[-1].gripping = False
            if enter:
                return



graphics = WormGraphics()

while 1:
    enemy_mode = True
    view_offset = 1200 / 7.0
    enemy_view_offset = 6.0 * 1200 / 7.0
    base_start_x = 1200
    enemy_width = 20
    start_x = base_start_x + view_offset
    min_x = base_start_x+view_offset+5*enemy_width

    ball_radius = 10.0
    segment_size = 17.0
    num_segments = 6
    ball_ball_friction = 0.0#0.4
    ball_ground_friction = 0.4
    ball_mass = 10.0
    spring_constant = 30.0

    new_level = generate_bar_level(5000)
    s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_ball_friction, ball_ground_friction, ball_mass, spring_constant))

    s.run(graphics)