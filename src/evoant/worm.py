from circular import Circular
from ball import Ball
import numpy as np

class Worm:
    def __init__(self, position, ball_radius, segment_size, num_segments, ball_friction):
        self.num_balls = num_segments + 1
        self.balls = [Ball(position + np.array([[i*segment_size],[0]]), ball_radius, ball_friction) for i in range(self.num_balls)]

    def get_x(self):
        return self.balls[0].get_x()

    def get_y(self):
        return self.balls[0].get_y()

    def step(self, level, gravity, delta_time):
        for i in range(self.num_balls):
            b = self.balls[i]
            b.bounce_on_level(level)
            # ~clamp?
            # for j in range(self.num_balls):
            #     if i != j:
            #         other = self.balls[j]
            #         distSq = np.dot(other.get_position()-b.get_position(), other.get_position()-b.get_position())
            #         if distSq < (other.radius+b.radius)*(other.radius+b.radius):
            #


            acceleration = gravity
            b.velocity += acceleration * delta_time
            b.position += b.velocity * delta_time

        return True







