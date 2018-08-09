from circular import Circular
import numpy as np

class Ball(Circular):
    def __init__(self, position, radius, friction, mass):
        Circular.__init__(self, position, radius, 12)
        self.velocity = np.array([[0.0], [0.0]])
        self.friction = friction
        self.bounced = False
        self.mass = mass

    def bounce_on_level(self, level):
        # if self.bounced:
        #     return

        bounce_direction = level.normal_direction_circular(self)
        if bounce_direction is not None:
            # print "BOUNCE"
            # print bounce_direction
            vel_component_in_bounce_direction = np.dot(self.velocity.T, bounce_direction)
            # (1) Remove all motion in bounce direction
            self.velocity += -vel_component_in_bounce_direction * bounce_direction
            # (2) Add motion opposite to bounce direction, modulated by friction
            self.velocity += -self.friction * vel_component_in_bounce_direction * bounce_direction



            self.position += 0.014*-vel_component_in_bounce_direction * bounce_direction
            i = 0
            while level.collides_with_circular(self) and i < 60:
                self.position += 0.014*-vel_component_in_bounce_direction * bounce_direction
                i+=1


            #TEST
            # self.velocity = vel_component_in_bounce_direction * bounce_direction

            self.bounced = True










