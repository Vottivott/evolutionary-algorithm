import numpy as np

from radar import Radar, ObjectRadar, ObjectAttributeRadar

BASE_STEP_SIZE = 40 #4
BASE_OBJECT_STEP_SIZE = 4

BASE_MAX_STEPS = int(250 * 4.0 / BASE_STEP_SIZE)

class RadarSystem:
    def __init__(self):
        max_steps = BASE_MAX_STEPS
        step_size = BASE_STEP_SIZE
        self.num_front_radars = 15
        directions = np.linspace(-3.0*np.pi/7, 3.0*np.pi/7, self.num_front_radars)
        self.radars = [Radar(dir, max_steps, step_size) for dir in directions]
        max_steps = BASE_MAX_STEPS
        step_size = -BASE_STEP_SIZE
        self.num_back_radars = 7
        directions = np.linspace(np.pi - 3.0 * np.pi / 7, np.pi + 3.0 * np.pi / 7, self.num_back_radars)
        self.radars.extend([Radar(dir, max_steps, step_size) for dir in directions])

        number_of_neurons = 20
        x_step_size = BASE_OBJECT_STEP_SIZE
        max_num_steps = 100
        max_dist = 1000
        only_left_half = False
        self.enemy_radar = ObjectRadar(number_of_neurons, x_step_size, max_num_steps, max_dist, only_left_half)


class FishRadarSystem:
    def __init__(self, mirrored):
        BASE_STEP_SIZE = 40#3  # 4
        BASE_OBJECT_STEP_SIZE = 4

        BASE_MAX_STEPS = int(250 * 2.0 / BASE_STEP_SIZE)

        USE_OPTIMIZED_VERSION = True

        max_steps = BASE_MAX_STEPS
        step_size = -BASE_STEP_SIZE
        self.num_front_radars = 2
        directions = np.linspace(-3.0*np.pi/4, 3.0*np.pi/4, self.num_front_radars)
        self.radars = [Radar(dir, max_steps, step_size, mirrored) for dir in directions]
        max_steps = BASE_MAX_STEPS
        step_size = BASE_STEP_SIZE
        self.num_back_radars = 2
        directions = np.linspace(np.pi - 3.0 * np.pi / 4, np.pi + 3.0 * np.pi / 4, self.num_back_radars)
        self.radars.extend([Radar(dir, max_steps, step_size, mirrored) for dir in directions])
        if USE_OPTIMIZED_VERSION:
            SQRT_TWO = 2 ** 0.5
            MAX_DIST = BASE_MAX_STEPS * BASE_STEP_SIZE
            if mirrored:
                self.radars[0].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(level.right_goal_x - position[0], level.bottom - position[1]))/MAX_DIST))
                self.radars[1].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(level.right_goal_x - position[0], position[1]-level.start_y))/MAX_DIST))
                self.radars[2].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(position[0]-level.left_goal_x - position[0], position[1] - level.start_y))/MAX_DIST))
                self.radars[3].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(position[0]-level.left_goal_x, level.bottom - position[1]))/MAX_DIST))
            else:
                self.radars[0].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(position[0]-level.left_goal_x, level.bottom - position[1]))/MAX_DIST))
                self.radars[1].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(position[0]-level.left_goal_x, position[1]-level.start_y))/MAX_DIST))
                self.radars[2].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(level.right_goal_x - position[0], position[1] - level.start_y))/MAX_DIST))
                self.radars[3].read = lambda position, level: (None, max(0,float(SQRT_TWO * min(level.right_goal_x - position[0], level.bottom - position[1]))/MAX_DIST))

        MAX_V_X = 30.0
        MAX_V_Y = 30.0
        number_of_neurons_per_vector = 8
        x_step_size = BASE_OBJECT_STEP_SIZE
        max_num_steps = 100
        max_dist = 1000
        only_left_half = False
        if mirrored:
            attribute_functions = [ lambda f: max(0.0, -1.0 * f.velocity[0]) / MAX_V_X,
                                    lambda f: max(0.0, -1.0 * -f.velocity[0]) / MAX_V_X,
                                    lambda f: max(0.0, f.velocity[1]) / MAX_V_Y,
                                    lambda f: max(0.0, -f.velocity[1]) / MAX_V_Y ]
        else:
            attribute_functions = [ lambda f: max(0.0, f.velocity[0]) / MAX_V_X,
                                    lambda f: max(0.0, -f.velocity[0]) / MAX_V_X,
                                    lambda f: max(0.0, f.velocity[1]) / MAX_V_Y,
                                    lambda f: max(0.0, -f.velocity[1]) / MAX_V_Y ]

        self.num_attr_radars = number_of_neurons_per_vector * (1 + len(attribute_functions))
        self.fish_radar = ObjectAttributeRadar(number_of_neurons_per_vector, x_step_size, max_num_steps, max_dist, only_left_half, attribute_functions, mirrored)
        self.fish_radar_num_dirs = number_of_neurons_per_vector


class EnemysRadarSystem:
    def __init__(self):
        max_steps = BASE_MAX_STEPS
        step_size = -BASE_STEP_SIZE
        self.num_front_radars = 15
        directions = np.linspace(-3.0*np.pi/7, 3.0*np.pi/7, self.num_front_radars)
        self.radars = [Radar(dir, max_steps, step_size) for dir in directions]
        max_steps = BASE_MAX_STEPS
        step_size = BASE_STEP_SIZE
        self.num_back_radars = 7
        directions = np.linspace(np.pi - 3.0 * np.pi / 7, np.pi + 3.0 * np.pi / 7, self.num_back_radars)
        self.radars.extend([Radar(dir, max_steps, step_size) for dir in directions])

        number_of_neurons = 10
        x_step_size = BASE_OBJECT_STEP_SIZE
        max_num_steps = 100
        max_dist = 1000
        only_left_half = True
        self.copter_radar = ObjectRadar(number_of_neurons, x_step_size, max_num_steps, max_dist, only_left_half)

        number_of_neurons = 10
        x_step_size = BASE_OBJECT_STEP_SIZE
        max_num_steps = 100
        max_dist = 800
        only_left_half = True
        self.shot_radar = ObjectRadar(number_of_neurons, x_step_size, max_num_steps, max_dist, only_left_half)

        number_of_neurons = 20
        x_step_size = BASE_OBJECT_STEP_SIZE
        max_num_steps = 100
        max_dist = 1000
        only_left_half = False
        self.enemy_radar = ObjectRadar(number_of_neurons, x_step_size, max_num_steps, max_dist, only_left_half) # for detecting other enemies



