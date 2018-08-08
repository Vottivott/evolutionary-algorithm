import numpy as np

from rectangular import Rectangular


class Smoke:
    def __init__(self, position, particle_rate, decay_rate, gravity, color, is_enemy=False):
        self.position = position
        self.particles = []
        self.frozen_particles = []
        self.particle_rate = particle_rate
        self.particle_sound_rate = 2 if is_enemy else 4
        self.decay_rate = decay_rate
        self.time = 0.0
        self.sound_time = 0.0
        self.particle_start_size = 5
        self.particle_end_size = 30
        self.shot_background_decay_rate = 0.3
        self.dive_background_decay = 0.6
        self.gravity = gravity
        self.color = color
        self.is_enemy = is_enemy

    def step(self, level, delta_time, firing):
        # return
        self.time += delta_time * self.particle_rate
        self.sound_time += delta_time * self.particle_sound_rate
        sputter = False
        while self.time > 1.0 and firing:
            self.create_particle()
            self.time -= 1

        while self.sound_time > 1.0 and firing:
            self.sound_time -= 1
            sputter = True

        for particle in list(self.particles):
            if not particle.step(level, delta_time):
                if particle.freeze_on_bounce:
                    self.frozen_particles.append(particle)
                self.particles.remove(particle)
        return sputter

    def create_particle(self):
        # return
        angle_range = 2.0/3*np.pi
        center_direction = 3*np.pi/2.0
        direction = (center_direction-angle_range/2) + angle_range * np.random.random()
        direction_vector = np.array([[np.cos(direction)],[-np.sin(direction)]])
        min_speed = 6.0
        speed_range = 5.0
        speed = min_speed + speed_range * np.random.random()
        velocity = direction_vector * speed
        p = SmokeParticle(self.position, velocity, self.decay_rate, self.particle_start_size, self.particle_end_size, np.array([[0.0],[0.0]]), False, self.color)
        self.particles.append(p)

    def create_explosion(self):
        if self.is_enemy:
            num_particles = 10
        else:
            num_particles = 40
        for i in range(num_particles):
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
        p = SmokeParticle(self.position, velocity, 0.0, size, size, self.gravity, True, self.color)
        self.particles.append(p)

    def create_shot_background(self, shot):
        for i in range(18):
            self.create_shot_background_particle(shot)

    def create_shot_background_particle(self, shot):
        angle_range = 0.015 * np.pi
        center_direction = 0
        direction = (center_direction - angle_range / 2) + angle_range * np.random.random()
        direction_vector = np.array([[np.cos(direction)], [-np.sin(direction)]])
        min_speed = 150.0
        speed_range = 35.0
        speed = min_speed + speed_range * np.random.random()
        velocity = direction_vector * speed
        p = SmokeParticle(self.position, velocity, self.shot_background_decay_rate, self.particle_start_size, self.particle_end_size, self.gravity, False, self.color, True, shot)
        self.particles.append(p)

    def create_dive_background(self):
        for i in range(10):
            self.create_dive_background_particle()



    def create_dive_background_particle(self):
        angle_range = 0.015 * np.pi
        center_direction = np.pi/2
        direction = (center_direction - angle_range / 2) + angle_range * np.random.random()
        direction_vector = np.array([[np.cos(direction)], [-np.sin(direction)]])
        min_speed = 30.0
        speed_range = 5.0
        speed = min_speed + speed_range * np.random.random()
        velocity = direction_vector * speed
        p = SmokeParticle(self.position, velocity, self.dive_background_decay, self.particle_start_size, self.particle_end_size, self.gravity, True, self.color)
        self.particles.append(p)




class SmokeParticle(Rectangular):
    def __init__(self, position, velocity, decay_rate, start_size, end_size, gravity, freeze_on_bounce, color, remove_on_bounce=False, shot=None):
        Rectangular.__init__(self, np.copy(position), start_size, start_size)
        self.velocity = np.array(velocity)
        self.alpha = 1.0
        self.decay_rate = decay_rate
        self.start_size = start_size
        self.expansion = end_size - start_size
        self.has_bounced = False # Allow only one bounce to prevent getting stuck in ground
        self.gravity = gravity
        self.freeze_on_bounce = freeze_on_bounce
        self.remove_on_bounce = remove_on_bounce
        self.color = color
        self.shot = shot

    def step(self, level, delta_time):
        self.alpha -= self.decay_rate * delta_time
        if not (self.freeze_on_bounce and self.has_bounced):
            acceleration = self.gravity
            # print self.velocity
            # print self.position
            # print
            self.velocity += acceleration * delta_time
            self.position += self.velocity * delta_time
            self.width = self.height = self.start_size + self.expansion*(1-self.alpha)
        if not self.has_bounced:
            bounce_direction = level.bounce_direction(self.velocity, self)
            if bounce_direction is not None:
                self.velocity = (bounce_direction * self.velocity.T.dot(self.velocity)**0.5)
                self.has_bounced = True
                if self.freeze_on_bounce:
                    return False
                if self.remove_on_bounce:
                    return False
        if self.shot is not None and self.shot.alpha < 0 and self.position[0] > self.shot.position[0]: # remove particles beyond the point wher the shot hit a wall
            return False
        if self.alpha < 0:
            return False
        return True


