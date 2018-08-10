from circular import Circular
from line_segment import LineSegment
import numpy as np
from math import ceil
from evomath import *

class Bounce():
    def __init__(self, normal, overlap, closest_edge):
        self.normal = normal
        self.overlap = overlap
        self.closest_edge = closest_edge
        self.planned_offset = np.array([[0.0],[0.0]])

    def __div__(self, other):
        return Bounce(self.normal / other, self.overlap / other, self.closest_edge / other)

class Ball(Circular):
    def __init__(self, position, radius, friction, mass):
        Circular.__init__(self, position, radius, 12)
        self.velocity = np.array([[0.0], [0.0]])
        self.friction = friction
        # self.bounced = False
        self.mass = mass

    # def bounce_on_line_segment(self, seg):
    #     dist = np.dot(seg.normal.T, self.position-seg.left)
    #     if dist < self.radius:
    #         print dist
    #         overlap = -(dist - self.radius)
    #
    #         vel_component_in_normal_direction = np.dot(self.velocity.T, seg.normal)
    #         # (1) Remove all motion in bounce direction
    #         self.velocity += -vel_component_in_normal_direction * seg.normal
    #         # (2) Add motion opposite to bounce direction, modulated by friction
    #         self.velocity += -self.friction * vel_component_in_normal_direction * seg.normal
    #         self.position += overlap * seg.normal

    def get_bounce_on_line_segment(self, seg):
        dist = np.dot(seg.normal.T, self.position - seg.left)
        if dist < self.radius:
            # print dist
            overlap = -(dist - self.radius)

            left_dot = np.dot(seg.tangent.T, self.position - seg.left)
            left_edge_dist = abs(left_dot)
            right_dot = np.dot(-seg.tangent.T, self.position - seg.right)
            right_edge_dist = abs(right_dot)
            if (left_edge_dist < right_edge_dist):
                closest_edge = seg.left
                if left_edge_dist >= self.radius and left_dot < 0.0:
                    return None
            else:
                closest_edge = seg.right
                if right_edge_dist >= self.radius and right_dot < 0.0:
                    return None


            return Bounce(seg.normal, overlap, closest_edge)
        else:
            return None



    def bounce_on_level(self, level):
        left = self.get_left()
        right = self.get_right()
        L = int(left / level.bar_width)
        R = int(ceil(right / level.bar_width))

        bounces = []

        for i in range(L, R):
            seg_ground = LineSegment(np.array([[i * level.bar_width], [level.ground[i]]]),
                              np.array([[(i + 1) * level.bar_width], [level.ground[i + 1]]]), False)
            seg_ceil = LineSegment(np.array([[i * level.bar_width], [level.ceiling[i]]]),
                              np.array([[(i + 1) * level.bar_width], [level.ceiling[i + 1]]]), True)
            bounce = self.get_bounce_on_line_segment(seg_ground)
            if bounce:
                bounces.append(bounce)
            bounce = self.get_bounce_on_line_segment(seg_ceil)
            if bounce:
                bounces.append(bounce)

        if len(bounces) == 0:
            return
        if len(bounces) == 1:
            final_bounce = bounces[0]

        else:
            point = mean([b.closest_edge for b in bounces])
            # Make up a bounce based on the common point instead of the line segments
            delta = self.position - point
            normal = normalized(delta)
            dist = np.linalg.norm(delta)
            overlap = max(0.0, self.radius - dist)
            # overlap = max(0.5, self.radius - dist) # TODO: Riktig losning
            final_bounce = Bounce(normal, overlap, point)


            #avg_normal = mean([b.normal for b in bounces])
            #avg_overlap = mean([b.overlap for b in bounces])
            #final_bounce = Bounce(normalized(avg_normal), avg_overlap)

        vel_component_in_normal_direction = np.dot(self.velocity.T, final_bounce.normal)
        # (1) Remove all motion in bounce direction
        self.velocity += -vel_component_in_normal_direction * final_bounce.normal
        # (2) Add motion opposite to bounce direction, modulated by friction
        self.velocity += -self.friction * vel_component_in_normal_direction * final_bounce.normal
        self.position += final_bounce.overlap * final_bounce.normal


    # def bounce_on_level(self, level):
    #     # if self.bounced:
    #     #     return
    #
    #     if self.get_bottom() > level.get_ground(self.get_x()):
    #         self.velocity[0] = 0.0
    #         self.velocity[1] = -30.0
    #     return
    #
    #     bounce_direction = level.normal_direction_circular(self)
    #     if bounce_direction is not None:
    #         # print "BOUNCE"
    #         # print bounce_direction
    #         vel_component_in_bounce_direction = np.dot(self.velocity.T, bounce_direction)
    #         # (1) Remove all motion in bounce direction
    #         self.velocity += -vel_component_in_bounce_direction * bounce_direction
    #         # (2) Add motion opposite to bounce direction, modulated by friction
    #         self.velocity += -self.friction * vel_component_in_bounce_direction * bounce_direction
    #
    #
    #
    #         self.position += 0.014*-vel_component_in_bounce_direction * bounce_direction
    #         i = 0
    #         while level.collides_with_circular(self) and i < 60:
    #             self.position += 0.014*-vel_component_in_bounce_direction * bounce_direction
    #             i+=1
    #
    #
    #         #TEST
    #         # self.velocity = vel_component_in_bounce_direction * bounce_direction
    #
    #         self.bounced = True










