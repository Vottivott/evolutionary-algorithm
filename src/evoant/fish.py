from evoant.ball import Ball
import numpy as np

from radar_system import FishRadarSystem

LIVING_COST = 0.001
MOVEMENT_COST = 0.001
BIRTH_COST = 0.4

AGEING_RATE = 0.0002

BIRTH_ENERGY_REQUIREMENT = 0.9


class Fish(Ball):
    def __init__(self, position, radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, mass, mirrored = False):
        Ball.__init__(self, position, radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, mass)
        self.energy = np.random.rand(1.0)#0.7
        self.age = np.random.rand(1.0)#0.0
        self.animation_velocity = None # only used for graphics
        self.radar_system = FishRadarSystem(mirrored)
        self.has_scored = False
        self.shoot_velocity = np.array([[0.0],[0.0]])
        self.do_shoot = False


    def step(self, delta_t):
        pass
        # self.energy -= LIVING_COST * delta_t
        # self.age += AGEING_RATE * delta_t

