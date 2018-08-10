from circular import Circular
from ball import Ball
from muscle import Muscle
import numpy as np
from evomath import *
from itertools import izip

class Worm:
    def __init__(self, position, ball_radius, segment_size, num_segments, ball_ball_friction, ball_ground_friction, ball_mass, spring_constant):
        self.num_balls = num_segments + 1
        self.balls = [Ball(position + np.array([[i*segment_size],[0]]), ball_radius, ball_ball_friction, ball_ground_friction, ball_mass) for i in range(self.num_balls)]
        self.muscles = [Muscle(b1, b2, segment_size, spring_constant) for b1,b2 in izip(self.balls[:-1],self.balls[1:])]
        self.max_y_velocity = 50.0
        self.max_x_velocity = 50.0
        self.max_real_muscle_length = 50.0
        self.muscle_flex_length = 13.0
        self.muscle_extend_length = 28.0

    def get_x(self):
        return self.balls[0].get_x()

    def get_y(self):
        return self.balls[0].get_y()

    def step(self, level, gravity, delta_time):

        # for b in self.balls:
        #     if abs(b.velocity[0]) > self.max_x_velocity:
        #         self.max_x_velocity = max(self.max_x_velocity, abs(b.velocity[0]))
        #         print self.max_x_velocity, self.max_y_velocity
        #     if abs(b.velocity[1]) > self.max_y_velocity:
        #         self.max_y_velocity = max(self.max_y_velocity, abs(b.velocity[1]))
        #         print self.max_x_velocity, self.max_y_velocity

        #TEST
        # if self.balls[0].bounced:
        #     self.muscles[0].target_length=50.0

        for b in self.balls:
            b.planned_offset = np.array([[0.0], [0.0]])





        for i in range(self.num_balls):
            b = self.balls[i]
            if not b.gripping:
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
                        if not b.gripping:
                            b.position += collisionLine * -margin / 2.0
                        if not other.gripping:
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

                        v1 *= b.ball_ball_friction
                        v2 *= other.ball_ball_friction

                        b.velocity += -u1Vector + collisionLine * v1
                        other.velocity += -u2Vector + collisionLine * v2

                        # print "bounce, %d, %d" % (i,j)
            import numpy.random
            # acceleration = numpy.random.permutation(np.array([[-1.0], [0.0]]))
            # acceleration = numpy.random.rand(1.0)*np.array([[1.0],[0.0]])


        for muscle in self.muscles:
            muscle.step(delta_time)

        for muscle in self.muscles:
            for b in self.balls:
                if b is not muscle.b1 and b is not muscle.b2:
                    muscle.collide_with_ball(b)

        for i,b in enumerate(self.balls):
            # if i>0:
            if not b.gripping:
                acceleration = gravity
                b.velocity += acceleration * delta_time
                b.velocity *= 0.975 # viscosity to prevent erratic movement from spring simulation
                b.position += b.velocity * delta_time
                b.position += b.planned_offset



        return True







