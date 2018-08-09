import numpy as np
from evomath import *

# Implementation of a muscle modeled as a spring, but where the neutral/unstretched length of the spring (target_length) can be changed dynamically

class Muscle:
    def __init__(self, b1, b2, target_length, spring_constant):
        self.b1 = b1
        self.b2 = b2
        self.target_length = target_length
        self.spring_constant = spring_constant

    def set_target_length(self, target_length):
        self.target_length = target_length

    def step(self, delta_time):
        # Small viscous force to prevent erratic behaviour
        # viscous_force = b1.

        d0 = self.target_length
        d = np.linalg.norm(self.b2.position - self.b1.position)
        delta = d - d0
        dir_from_b1_to_b2 = normalized(self.b2.position - self.b1.position)
        spring_force = -self.spring_constant * delta * dir_from_b1_to_b2
        b1_acceleration = spring_force * -1.0 / self.b1.mass
        b2_acceleration = spring_force * 1.0 / self.b2.mass
        self.b1.velocity += b1_acceleration * delta_time
        self.b2.velocity += b2_acceleration * delta_time

