from rectangular import Rectangular
import numpy as np

class Copter(Rectangular):
    def __init__(self, position, size):
        Rectangular.__init__(self, position, size, size)
        self.velocity = np.array([[10.0],[0.0]])

    def step(self, level, gravity, fire_force, delta_time):
        self.acceleration = gravity + fire_force
        self.velocity += self.acceleration * delta_time
        self.position += self.velocity * delta_time
        if level.collides_with(self):
            return False
        return True



