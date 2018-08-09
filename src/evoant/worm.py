from circular import Circular
from ball import Ball
import numpy as np
from evomath import *

class Worm:
    def __init__(self, position, ball_radius, segment_size, num_segments, ball_friction, ball_mass):
        self.num_balls = num_segments + 1
        self.balls = [Ball(position + np.array([[i*segment_size],[0]]), ball_radius, ball_friction, ball_mass) for i in range(self.num_balls)]

    def get_x(self):
        return self.balls[0].get_x()

    def get_y(self):
        return self.balls[0].get_y()

    def step(self, level, gravity, delta_time):

        for i in range(self.num_balls):
            b = self.balls[i]
            b.bounce_on_level(level)

        for i in range(self.num_balls):
            b = self.balls[i]
            for j in range(self.num_balls):
                if i != j:
                    other = self.balls[j]
                    distSq = np.dot((other.get_position()-b.get_position()).T, other.get_position()-b.get_position())
                    if distSq < (other.radius+b.radius)*(other.radius+b.radius):
                        # If collision
                        collisionLine = normalized(other.position - b.position)
                        dist = distSq**0.5
                        margin = b.radius + other.radius - dist

                        # Move to contact point (moving both balls the same distance)
                        b.position += collisionLine * -margin / 2.0
                        other.position += collisionLine * margin / 2.0

                        u1Vector = b.velocity.T.dot(collisionLine) * collisionLine
                        u2Vector = other.velocity.T.dot(collisionLine) * collisionLine
                        u1 = collisionLine.T.dot(u1Vector)
                        u2 = collisionLine.T.dot(u2Vector)
                        m1 = b.mass
                        m2 = other.mass
                        I = m1*u1 + m2*u2
                        R = -(u2 - u1)
                        v1 = (I - m2*R) / (m1 + m2)
                        v2 = R + v1

                        v1 *= b.friction
                        v2 *= other.friction

                        b.velocity += -u1Vector + collisionLine * v1
                        other.velocity += -u2Vector + collisionLine * v2

                        # print "bounce, %d, %d" % (i,j)
            import numpy.random
            # acceleration = numpy.random.permutation(np.array([[-1.0], [0.0]]))
            # acceleration = numpy.random.rand(1.0)*np.array([[1.0],[0.0]])
            acceleration = gravity
            b.velocity += acceleration * delta_time
            b.position += b.velocity * delta_time

        return True







