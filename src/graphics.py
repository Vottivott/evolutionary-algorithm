from itertools import izip

import pygame, sys
from pygame.locals import *
import numpy as np
import pyglet.media
# import pyglet
class Graphics:
    def __init__(self):
        pygame.init()
        self.size = w, h = 1200, 500
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.view_offset = w/7#17.0
        self.sputter_sound = pyglet.media.StaticSource(pyglet.media.load('sputter_sound.wav'))#pygame.mixer.Sound('beep.wav')
        self.sound_player = pyglet.media.Player()
        self.crash_sound = pyglet.media.StaticSource(pyglet.media.load('crash_sound.wav'))#pyglet.media.load('crash_sound.wav')
        # self.crash_sound = pyglet.resource.media('crash_sound.wav')#pygame.mixer.Sound('crash_sound.wav')
        # effect.play()

    def update(self, copter_simulation):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            # if event.type == KEYDOWN and event.key == K_t:
            #     trace = not trace

        self.screen.fill((240, 245, 250))
        s = pygame.Surface(self.size)  # the size of your rect
        # if trace:
        #     s.set_alpha(28)  # alpha level
        # s.fill((0, 0, 0))  # this fills the entire surface
        # self.screen.blit(s, (0, 0))  # (0,0) are the top-left coordinates


        #     pointlists = get_curve()
        #     color = min(np.random.normal(200, 30, 1), 255)

        self.draw_smoke(copter_simulation)
        self.draw_copter(copter_simulation.copter)
        self.draw_level(copter_simulation)
        if False:
            self.draw_radars(copter_simulation)

        pygame.display.flip()
        self.clock.tick(60)

        keys = pygame.key.get_pressed()  # checking pressed keys
        return keys[pygame.K_SPACE]

    def draw_level(self, copter_simulation):
        level = copter_simulation.level
        cx = copter_simulation.copter.get_x()
        start = cx - self.view_offset
        end = start + self.size[0]
        start = min(len(level), max(0, start))
        end = min(len(level), max(0, end))
        start = int(start)
        end = int(end)
        ceiling_region = copter_simulation.level.ceiling[start:end]
        ground_region = copter_simulation.level.ground[start:end]
        ceiling_coords = [(x, y) for x, y in enumerate(list(ceiling_region))]
        ceiling_pointlist = [(this, next) for this, next in izip(ceiling_coords, ceiling_coords[1:])]
        ground_coords = [(x, y) for x, y in enumerate(list(ground_region))]
        ground_pointlist = [(this, next) for this, next in izip(ground_coords, ground_coords[1:])]
        c_cave = (150,160,170)
        # for point in ceiling_pointlist:
        #     pygame.draw.lines(self.screen, (255, color, 0), False, point, 2)
        # for point in ground_pointlist:
        #     pygame.draw.lines(self.screen, (255, color, 0), False, point, 2)
        pygame.draw.polygon(self.screen, c_cave, [[0,0]] + ceiling_coords + [[self.size[0],0]])
        pygame.draw.polygon(self.screen, c_cave, [[0,self.size[1]]] + ground_coords + [[self.size[0],self.size[1]]])

    def play_crash_sound(self):
        # self.sound_player.queue(self.crash_sound)
        self.crash_sound.play()

    def play_sputter_sound(self):
        self.sputter_sound.play()
        # self.sputter_sound.play()

    def draw_copter(self, copter):
        if not copter.exploded:
            pygame.draw.rect(self.screen, (175,20,0), pygame.Rect(self.view_offset-copter.width/2.0, copter.get_top(),    copter.width, copter.height))

    def draw_smoke(self, copter_simulation):
        for particle in copter_simulation.smoke.particles:
            self.draw_smoke_particle(particle, copter_simulation)

    def np_to_screen_coord(self, np_vector, copter_simulation):
        x = np_vector[0] - copter_simulation.copter.get_x()
        y = np_vector[1]
        return (self.view_offset + x, y)

    def draw_smoke_particle(self, particle, copter_simulation):
        x = particle.get_left() - copter_simulation.copter.get_x()
        y = particle.get_top()
        s = pygame.Surface((particle.width, particle.height))  # the size of your rect
        s.set_alpha(int(particle.alpha * 255))
        s.fill((200, 0, 0))
        self.screen.blit(s, (self.view_offset + x, y))
        #pygame.draw.rect(self.screen, (255,180,0), pygame.Rect(self.view_offset + x, y, particle.width, particle.height))

    def draw_radars(self, copter_simulation):
        if not copter_simulation.copter.exploded:
            for radar in copter_simulation.radar_system.radars:
                point, dist = radar.point, radar.dist
                if (point,dist) == (None,None):
                    point,dist = radar.read(copter_simulation.copter.position, copter_simulation.level)
                if dist < 1:
                    dist = 0.5 + dist/2 # for drawing only
                    color = (255, int(150+(255-150)*dist), int(dist*255))
                    pygame.draw.circle(self.screen, color, self.np_to_screen_coord(point, copter_simulation), 5)

