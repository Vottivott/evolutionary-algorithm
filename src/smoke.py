import numpy as np

from rectangular import Rectangular


class Smoke:
    def __init__(self, position, particle_rate, decay_rate, gravity):
        self.position = position
        self.particles = []
        self.particle_rate = particle_rate
        self.decay_rate = decay_rate
        self.time = 0.0
        self.particle_start_size = 5
        self.particle_end_size = 30
        self.gravity = gravity

    def step(self, level, delta_time, firing):
        self.time += delta_time * self.particle_rate
        while self.time > 1.0 and firing:
            self.create_particle()
            self.time -= 1

        for particle in list(self.particles):
            if not particle.step(level, delta_time):
                self.particles.remove(particle)

    def create_particle(self):
        angle_range = 2.0/3*np.pi
        center_direction = 3*np.pi/2.0
        direction = (center_direction-angle_range/2) + angle_range * np.random.random()
        direction_vector = np.array([[np.cos(direction)],[-np.sin(direction)]])
        min_speed = 6.0
        speed_range = 5.0
        speed = min_speed + speed_range * np.random.random()
        velocity = direction_vector * speed
        p = SmokeParticle(self.position, velocity, self.decay_rate, self.particle_start_size, self.particle_end_size, np.array([[0.0],[0.0]]), False)
        self.particles.append(p)

    def create_explosion(self):
        for i in range(40):
            self.create_explosion_particle()

    def create_explosion_particle(self):
        angle_range = 2.0 * np.pi
        center_direction = 3 * np.pi / 2.0
        direction = (center_direction - angle_range / 2) + angle_range * np.random.random()
        direction_vector = np.array([[np.cos(direction)], [-np.sin(direction)]])
        min_speed = 9.0
        speed_range = 7.0
        speed = min_speed + speed_range * np.random.random()
        velocity = direction_vector * speed
        size = 10
        p = SmokeParticle(self.position, velocity, 0.0, size, size, self.gravity, True)
        self.particles.append(p)


class SmokeParticle(Rectangular):
    def __init__(self, position, velocity, decay_rate, start_size, end_size, gravity, freeze_on_bounce):
        Rectangular.__init__(self, np.copy(position), start_size, start_size)
        self.velocity = np.array(velocity)
        self.alpha = 1.0
        self.decay_rate = decay_rate
        self.start_size = start_size
        self.expansion = end_size - start_size
        self.has_bounced = False # Allow only one bounce to prevent getting stuck in ground
        self.gravity = gravity
        self.freeze_on_bounce = freeze_on_bounce

    def step(self, level, delta_time):
        if not (self.freeze_on_bounce and self.has_bounced):
            acceleration = self.gravity
            self.velocity += acceleration * delta_time
            self.position += self.velocity * delta_time
        self.alpha -= self.decay_rate * delta_time
        self.width = self.height = self.start_size + self.expansion*(1-self.alpha)
        if not self.has_bounced:
            bounce_direction = level.bounce_direction(self.velocity, self)
            if bounce_direction is not None:
                self.velocity = (bounce_direction * self.velocity.T.dot(self.velocity)**0.5)
                self.has_bounced = True


        if self.alpha < 0:
            return False
        return True


