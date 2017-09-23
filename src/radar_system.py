import numpy as np

from radar import Radar


class RadarSystem:
    def __init__(self):
        max_steps = 250
        step_size = 4
        self.num_front_radars = 15
        directions = np.linspace(-3.0*np.pi/7, 3.0*np.pi/7, self.num_front_radars)
        self.radars = [Radar(dir, max_steps, step_size) for dir in directions]
        max_steps = 250
        step_size = -4
        self.num_back_radars = 7
        directions = np.linspace(np.pi - 3.0 * np.pi / 7, np.pi + 3.0 * np.pi / 7, self.num_back_radars)
        self.radars.extend([Radar(dir, max_steps, step_size) for dir in directions])


class EnemysRadarSystem:
    def __init__(self):
        max_steps = 250
        step_size = -4
        self.num_front_radars = 15
        directions = np.linspace(-3.0*np.pi/7, 3.0*np.pi/7, self.num_front_radars)
        self.radars = [Radar(dir, max_steps, step_size) for dir in directions]
        max_steps = 250
        step_size = 4
        self.num_back_radars = 7
        directions = np.linspace(np.pi - 3.0 * np.pi / 7, np.pi + 3.0 * np.pi / 7, self.num_back_radars)
        self.radars.extend([Radar(dir, max_steps, step_size) for dir in directions])

