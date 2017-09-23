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
        self.enemy_view_offset = 6.0*w/7
        self.sputter_sound = pyglet.media.StaticSource(pyglet.media.load('sputter_sound.wav'))#pygame.mixer.Sound('beep.wav')
        self.sound_player = pyglet.media.Player()
        self.crash_sound = pyglet.media.StaticSource(pyglet.media.load('crash_sound.wav'))#pyglet.media.load('crash_sound.wav')
        self.shot_sound = pyglet.media.StaticSource(pyglet.media.load('shot_sound.wav'))
        self.enemy_hit_sound = pyglet.media.StaticSource(pyglet.media.load('enemy_hit_sound.wav'))
        self.enemy_sputter_sound = pyglet.media.StaticSource(pyglet.media.load('enemy_sputter_sound.wav'))
        self.enemy_dive_sound = pyglet.media.StaticSource(pyglet.media.load('enemy_dive_sound.wav'))
        self.main_copter_smoke_color = (200, 0, 0)
        self.main_copter_color = (175,20,0)
        self.enemy_smoke_color = (40, 40, 40)
        self.enemy_color = (30, 30, 30)
        self.who_to_follow = True # True = main robot, number n = enemy with index n
        # TEST
        # self.main_copter_color = self.enemy_color
        # self.main_copter_smoke_color = self.enemy_smoke_color
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
        # s = pygame.Surface(self.size)  # the size of your rect
        # if trace:
        #     s.set_alpha(28)  # alpha level
        # s.fill((0, 0, 0))  # this fills the entire surface
        # self.screen.blit(s, (0, 0))  # (0,0) are the top-left coordinates


        #     pointlists = get_curve()
        #     color = min(np.random.normal(200, 30, 1), 255)

        self.draw_smoke(copter_simulation)
        for i in range(len(copter_simulation.enemies)):
            self.draw_enemy_smoke(copter_simulation, i)
        self.draw_copter(copter_simulation.copter, copter_simulation)
        for enemy in copter_simulation.enemies:
            self.draw_enemy(enemy, copter_simulation)
        self.draw_level(copter_simulation)
        if True:
            self.draw_radars(copter_simulation)

        # self.draw_shots(copter_simulation)

        pygame.display.flip()
        self.clock.tick(60)

        keys = pygame.key.get_pressed()  # checking pressed keys
        return keys[pygame.K_SPACE], (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]), keys[pygame.K_LEFT]

    def draw_level(self, copter_simulation):
        level = copter_simulation.level
        if self.who_to_follow == 'main':
            cx = copter_simulation.copter.get_x()
        else:
            cx = copter_simulation.enemies[self.who_to_follow].get_x()
        start = cx - self.get_view_offset()
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

    def play_enemy_sputter_sound(self):
        self.enemy_sputter_sound.play()

    def play_enemy_dive_sound(self):
        self.enemy_dive_sound.play()


    def play_shot_sound(self):
        self.shot_sound.play()

    def play_enemy_hit_sound(self):
        self.enemy_hit_sound.play()


    def draw_copter(self, copter, copter_simulation):
        if not copter.exploded:
            pos = self.np_to_screen_coord(copter.position, copter_simulation)
            pygame.draw.rect(self.screen, self.main_copter_color, pygame.Rect(pos[0]-copter.width/2.0, pos[1]-copter.height/2.0,    copter.width, copter.height))

    def draw_enemy(self, enemy, copter_simulation):
        if not enemy.exploded:
            pos = self.np_to_screen_coord(enemy.position, copter_simulation)
            pygame.draw.rect(self.screen, self.enemy_color, pygame.Rect(pos[0]-enemy.width/2.0, pos[1]-enemy.height/2.0, enemy.width, enemy.height))


    def draw_smoke(self, copter_simulation):
        for particle in copter_simulation.smoke.particles:
            self.draw_smoke_particle(particle, copter_simulation)

    def draw_enemy_smoke(self, copter_simulation, index):
        smoke = copter_simulation.enemy_smokes[index]
        for particle in smoke.particles:
            self.draw_smoke_particle(particle, copter_simulation)

    def np_to_screen_coord(self, np_vector, copter_simulation):
        if self.who_to_follow == 'main': # follow main robot
            x = np_vector[0] - copter_simulation.copter.get_x()
            y = np_vector[1]
            view_offset = self.view_offset
        else:
            x = np_vector[0] - copter_simulation.enemies[self.who_to_follow].get_x()
            y = np_vector[1]
            view_offset = self.enemy_view_offset
        return (view_offset + x, y)

    def get_view_offset(self):
        return self.view_offset if self.who_to_follow == 'main' else self.enemy_view_offset

    def draw_smoke_particle(self, particle, copter_simulation):

        # x = particle.get_left() - copter_simulation.copter.get_x()
        # y = particle.get_top()
        s = pygame.Surface((particle.width, particle.height))  # the size of your rect
        s.set_alpha(int(particle.alpha * 255))
        s.fill(particle.color)
        # self.screen.blit(s, (self.view_offset + x, y))
        x,y = self.np_to_screen_coord(particle.position, copter_simulation)
        pos  = (x - particle.width/2, y - particle.height/2)
        self.screen.blit(s, pos)
        #pygame.draw.rect(self.screen, (255,180,0), pygame.Rect(self.view_offset + x, y, particle.width, particle.height))

    def draw_radars(self, copter_simulation):
        if not copter_simulation.copter.exploded:
            radar_systems = [copter_simulation.radar_system]
            for radar_system in radar_systems:
                for radar in radar_system.radars:
                    point, dist = radar.point, radar.dist
                    if True:#(point,dist) == (None,None):
                        point,dist = radar.read(copter_simulation.copter.position, copter_simulation.level)
                    if True:#dist < 1:
                        dist = 0.5 + dist/2 # for drawing only
                        color = (255, int(150+(255-150)*dist), int(dist*255))
                        pygame.draw.circle(self.screen, color, self.np_to_screen_coord(point, copter_simulation), 5)

    def draw_shots(self, copter_simulation):
        for shot in copter_simulation.shots:
            x,y  = self.np_to_screen_coord(shot.position, copter_simulation)
            x = x - shot.width / 2.0
            y = y - shot.height/ 2.0
            pygame.draw.rect(self.screen, (255, 180, 0),
                             pygame.Rect(x, y, shot.width, shot.height))



