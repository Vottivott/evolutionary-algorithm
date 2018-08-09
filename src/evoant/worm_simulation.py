import numpy as np

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
from level import generate_level
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
            graphics.update(self)



graphics = WormGraphics()

enemy_mode = True
view_offset = 1200 / 7.0
enemy_view_offset = 6.0 * 1200 / 7.0
base_start_x = 1200
enemy_width = 20
start_x = base_start_x + view_offset
min_x = base_start_x+view_offset+5*enemy_width

ball_radius = 15.0
segment_size = 40.0
num_segments = 4
ball_friction = 0.8
ball_mass = 10.0

new_level = generate_level(5000)
s = WormSimulation(new_level, Worm(np.array([[start_x], [new_level.y_center(start_x)]]), ball_radius, segment_size, num_segments, ball_friction, ball_mass))

s.run(graphics)