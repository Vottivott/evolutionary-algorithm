from rectangular import Rectangular
import numpy as np


class Shot(Rectangular):
    def __init__(self, position, gravity):
        Rectangular.__init__(self, np.copy(position), 50, 10)
        # self.position[1] += -23/2.0
        self.velocity = np.array([[167.5],[0]])
        self.gravity = gravity
        self.alpha = 1.0
        self.decay_rate = 0.35

    def step(self, level, delta_time):
        self.velocity += self.gravity * delta_time
        self.position += self.velocity * delta_time
        self.alpha -= self.decay_rate * delta_time
        if level.collides_with_multipoint(self):
            return False
        if self.alpha < 0:
            return False
        return True


