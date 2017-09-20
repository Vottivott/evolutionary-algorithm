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
        self.gravity = np.array([[0.0],[9.8]])
        self.delta_t = 1.0/4

    def run(self, graphics=None):
        if graphics:
            self.smoke = Smoke(self.copter.position, 4, 0.05)
        while 1:
            fire_force = np.array([[0.0],[-9.8]])
            still_flying = self.copter.step(self.level, self.gravity, fire_force, self.delta_t)
            # if not still_flying:
            #     print "!"
            if graphics:
                self.smoke.step(self.level, self.delta_t)
                graphics.update(self)

if __name__ == "__main__":
    graphics = Graphics()
    s = CopterSimulation(generate_level(10000), Copter(np.array([[0.0],[150.0]]), 20))
    s.run(graphics)