from evoant.ball import Ball
import numpy as np

LIVING_COST = 0.001
MOVEMENT_COST = 0.001
BIRTH_COST = 0.4

AGEING_RATE = 0.0002

BIRTH_ENERGY_REQUIREMENT = 0.9


class Fish(Ball):
    def __init__(self, position, radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, mass):
        Ball.__init__(self, position, radius, ball_ball_restitution, ball_ground_restitution, ball_ground_friction, mass)
        self.energy = np.random.rand(1.0)#0.7
        self.age = np.random.rand(1.0)#0.0
        self.animation_velocity = None # only used for graphics

    def step(self, delta_t):
        self.energy -= LIVING_COST * delta_t
        self.age += AGEING_RATE * delta_t
