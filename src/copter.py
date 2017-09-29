from rectangular import Rectangular
import numpy as np

class Copter(Rectangular):
    def __init__(self, position, size):
        Rectangular.__init__(self, position, size, size)
        self.base_velocity = np.array([[10.0], [0.0]])
        self.velocity = np.array([[10.0], [0.0]])
        self.exploded = False
        self.firing = False
        self.max_y_velocity = 20.0
        self.max_x_velocity = 180.0
        self.HSPEED_REQUIRED_TO_SHOOT = -20

    def step(self, level, gravity, fire_force, delta_time):
        acceleration = gravity + fire_force
        self.velocity += acceleration * delta_time
        if not self.exploded:
            self.velocity[0] += (self.base_velocity[0] - self.velocity[0]) * 0.4 * delta_time
        # self.velocity[0] = max(-self.max_x_velocity, min(self.max_x_velocity, self.velocity[0]))
        self.velocity[1] = max(-self.max_y_velocity, min(self.max_y_velocity, self.velocity[1]))
        self.position += self.velocity * delta_time
        if self.exploded:
            self.velocity *= 0.97
        if level.collides_with(self):
            return False
        return True

    def recoil(self):
        self.velocity[0] -= 128





