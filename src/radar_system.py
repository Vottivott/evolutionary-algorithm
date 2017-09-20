import numpy as np

from radar import Radar


class RadarSystem:
    def __init__(self):
        max_steps = 250
        step_size = 4
        directions = np.linspace(-3.0*np.pi/7, 3.0*np.pi/7, 15)
        self.radars = [Radar(dir, max_steps, step_size) for dir in directions]