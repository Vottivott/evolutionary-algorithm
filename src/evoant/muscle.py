import numpy as np
from evomath import *
from line_segment import LineSegment

# Implementation of a muscle modeled as a spring, but where the neutral/unstretched length of the spring (target_length) can be changed dynamically

class Muscle:
    def __init__(self, b1, b2, target_length, spring_constant):
        self.b1 = b1
        self.b2 = b2
        self.target_length = target_length
        self.spring_constant = spring_constant
        self.line_segment = None

    def set_target_length(self, target_length):
        self.target_length = target_length

    def step(self, delta_time):
        # Small viscous force to prevent erratic behaviour
        # viscous_force = b1.

        d0 = self.target_length
        d = np.linalg.norm(self.b2.position - self.b1.position)
        delta = d - d0
        self.line_segment = LineSegment(self.b1.position, self.b2.position, False)
        dir_from_b1_to_b2 = normalized(self.b2.position - self.b1.position)
        spring_force = -self.spring_constant * delta * dir_from_b1_to_b2
        b1_acceleration = spring_force * -1.0 / self.b1.mass
        b2_acceleration = spring_force * 1.0 / self.b2.mass
        if not self.b1.gripping:
            self.b1.velocity += b1_acceleration * delta_time
        if not self.b2.gripping:
            self.b2.velocity += b2_acceleration * delta_time

    def collide_with_ball(self, ball):
        seg = self.line_segment
        dist = np.dot(seg.normal.T, ball.position - seg.left)
        if abs(dist) < ball.radius:
            if dist < 0:
                overlap = -(-dist - ball.radius)
            else:
                overlap = -(dist - ball.radius)
            left_dot = np.dot(seg.tangent.T, ball.position - seg.left)
            left_edge_dist = abs(left_dot)
            right_dot = np.dot(-seg.tangent.T, ball.position - seg.right)
            right_edge_dist = abs(right_dot)
            if (left_edge_dist < right_edge_dist):
                closest_edge = seg.left
                if left_edge_dist >= ball.radius and left_dot < 0.0:
                    return None
            else:
                closest_edge = seg.right
                if right_edge_dist >= ball.radius and right_dot < 0.0:
                    return None
            if dist < 0:
                ball.position -= overlap * seg.normal
            else:
                ball.position += overlap * seg.normal



