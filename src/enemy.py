from rectangular import Rectangular
import numpy as np

class Enemy(Rectangular):
    max_x_velocity = 13.5
    max_y_velocity = 32.0
    def __init__(self, position):
        size = 20
        Rectangular.__init__(self, position, size, size)
        self.base_velocity = np.array([[0.0], [0.0]])
        self.velocity = np.array([[0.0], [0.0]])
        self.exploded = False
        self.firing = False
        self.diving = False
        self.moving_left = False
        self.moving_left_force = np.array([[0.3 * -20], [0.0]])
        self.collision_friction = 0.3
        self.MIN_TIME_BETWEEN_DIVES = 10
        self.time_since_last_dive = 0

    def step(self, level, gravity, fire_force, delta_time):
        acceleration = gravity + fire_force + self.moving_left * self.moving_left_force
        self.velocity += acceleration * delta_time
        # if not self.exploded:
        self.velocity[0] += (self.base_velocity[0] - self.velocity[0]) * 0.4 * delta_time
        self.velocity[0] = max(-Enemy.max_x_velocity, min(Enemy.max_x_velocity, self.velocity[0]))
        self.velocity[1] = max(-Enemy.max_y_velocity, min(Enemy.max_y_velocity, self.velocity[1]))
        self.position += self.velocity * delta_time
        self.time_since_last_dive += 1
        if self.exploded:
            self.velocity *= 0.97
        if level.collides_with_rectangular(self):
            return False
        return True

    def dive(self):
        if self.time_since_last_dive > self.MIN_TIME_BETWEEN_DIVES:
            self.velocity[1] += 20
            self.time_since_last_dive = 0
            return True
        else:
            return False





