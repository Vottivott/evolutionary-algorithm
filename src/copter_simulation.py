import numpy as np

from copter import Copter
from graphics import Graphics
from level import generate_level
from smoke import Smoke


class CopterSimulation:
    def __init__(self, level, copter):
        self.level = level
        self.copter = copter
        self.smoke = None
        self.gravity = np.array([[0.0],[0.2*9.8]])
        self.delta_t = 1.0/4
        self.space_pressed = False
        self.force_when_fire_is_on = np.array([[0.0],[0.2*-20]])
        self.force_when_fire_is_off = np.array([[0.0],[0.0]])

    def run(self, graphics=None, user_control=False):
        if graphics:
            self.smoke = Smoke(self.copter.position, 4, 0.05, self.gravity)
        while 1:
            if self.copter.velocity[0] < 0.1:
                return True
            if graphics and user_control:
                firing = self.space_pressed
            else:
                firing = True
            if firing:
                fire_force = self.force_when_fire_is_on
            else:
                fire_force = self.force_when_fire_is_off
            still_flying = self.copter.step(self.level, self.gravity, fire_force, self.delta_t)
            if not still_flying:
                if not self.copter.exploded:
                    self.smoke.create_explosion()
                    self.copter.exploded = True
                firing = False
            if graphics:
                self.smoke.step(self.level, self.delta_t, firing)
                self.space_pressed = graphics.update(self)


if __name__ == "__main__":
    graphics = Graphics()
    while 1:
        level = generate_level(10000)
        s = CopterSimulation(level, Copter(np.array([[graphics.view_offset], [level.y_center(graphics.view_offset)]]), 20))
        s.run(graphics, user_control=True)