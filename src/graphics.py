from itertools import izip

import pygame, sys
from pygame.locals import *
import numpy as np
import pyglet.media

def tuple_add(a,b):
    return [v_a+v_b for v_a,v_b in zip(a,b)]

def tuple_scale(a,factor):
    return [v_a*factor for v_a in a]


def to_tuple(np_array):
    try:
        return [np.asscalar(n) for n in np_array]
    except TypeError:
        return np_array

class ScoreText:
    def __init__(self, hitPosition):
        self.position = np.copy(hitPosition)
        self.time_left = 90
        self.rise_rate = 1

    def update(self):
        self.position[1] -= self.rise_rate
        self.time_left -= 1
        if self.time_left <= 0:
            return False
        return True

# import pyglet
class Graphics:
    def __init__(self):
        pygame.init()
        pygame.font.init()
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
        self.who_to_follow = 'main' # True = main robot, number n = enemy with index n
        self.user_control = None
        # TEST
        # self.main_copter_color = self.enemy_color
        # self.main_copter_smoke_color = self.enemy_smoke_color
        # self.crash_sound = pyglet.resource.media('crash_sound.wav')#pygame.mixer.Sound('crash_sound.wav')
        # effect.play()
        self.show_enemy_radars = False
        self.show_enemy_object_radars = False
        self.show_copter_radars = False
        self.show_copter_object_radars = False
        self.player_box_timer = 0.0
        self.PLAYER_BOX_TIME = 60

        self.score = 0
        self.score_font = pygame.font.Font('Steppes.ttf', 30)
        self.shot_score_font = pygame.font.Font('Steppes.ttf', 50)
        self.shot_score_color = (255,255,255)#self.main_copter_color
        self.shot_score_surface = self.score_font.render('+1000', False, self.shot_score_color)
        self.shot_score_texts = []

        self.frame = 0



    def update(self, copter_simulation):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_e:
                    self.show_enemy_radars = not self.show_enemy_radars
                if event.key == K_c:
                    self.show_copter_radars = not self.show_copter_radars
                if event.key == K_r:
                    self.show_enemy_object_radars = not self.show_enemy_object_radars
                if event.key == K_v:
                    self.show_copter_object_radars = not self.show_copter_object_radars
                if event.key == K_TAB:
                    if not (pygame.key.get_mods() & KMOD_SHIFT):
                        if self.user_control is None:
                            if len(copter_simulation.enemy_instances):
                                self.user_control = 0
                                self.player_box_timer = self.PLAYER_BOX_TIME
                            else:
                                self.user_control = 'main'
                        elif self.user_control == 'main':
                            self.user_control = None
                            self.player_box_timer = self.PLAYER_BOX_TIME
                        else:
                            self.user_control += 1
                            if self.user_control >= len(copter_simulation.enemy_instances):
                                self.user_control = None
                            else:
                                self.player_box_timer = self.PLAYER_BOX_TIME
                    else:
                        if self.user_control is None:
                            self.user_control = 'main'
                        elif self.user_control == 'main':
                            if len(copter_simulation.enemy_instances):
                                self.user_control = len(copter_simulation.enemy_instances) - 1
                                self.player_box_timer = self.PLAYER_BOX_TIME
                            else:
                                self.user_control = None
                        else:
                            self.user_control -= 1
                            if self.user_control < 0 or self.user_control >= len(copter_simulation.enemy_instances):
                                self.user_control = 'main'
                            else:
                                self.player_box_timer = self.PLAYER_BOX_TIME


        self.screen.fill((240, 245, 250))
        # s = pygame.Surface(self.size)  # the size of your rect
        # if trace:
        #     s.set_alpha(28)  # alpha level
        # s.fill((0, 0, 0))  # this fills the entire surface
        # self.screen.blit(s, (0, 0))  # (0,0) are the top-left coordinates


        #     pointlists = get_curve()
        #     color = min(np.random.normal(200, 30, 1), 255)

        self.draw_smoke(copter_simulation)
        for i in range(len(copter_simulation.enemy_instances)):
            self.draw_enemy_smoke(copter_simulation, i)
        self.draw_copter(copter_simulation.copter, copter_simulation)
        for ei in copter_simulation.enemy_instances:
            self.draw_enemy(ei.enemy, copter_simulation)
        self.draw_level(copter_simulation)
        if self.show_enemy_radars:
            self.draw_enemies_radars(copter_simulation)
        if self.show_copter_radars:
            self.draw_radars(copter_simulation)
        if self.show_copter_object_radars:
            self.draw_object_radars(copter_simulation)
        if self.show_enemy_object_radars:
            self.draw_enemy_object_radars(copter_simulation)

        if self.user_control == 'main':
            if not copter_simulation.copter.exploded and self.player_box_timer > 0:
                self.draw_player_box(copter_simulation.copter, copter_simulation)
                self.player_box_timer -= 1
                #print self.player_box_timer
        elif self.user_control is not None:
            if self.user_control < len(copter_simulation.enemy_instances):
                if not copter_simulation.enemy_instances[self.user_control].enemy.exploded and self.player_box_timer > 0:
                    self.draw_player_box(copter_simulation.enemy_instances[self.user_control].enemy, copter_simulation)
                    self.player_box_timer -= 1

        # self.draw_shots(copter_simulation)

        self.draw_score(copter_simulation)

        self.update_shot_score_texts()
        self.draw_shot_score_texts(copter_simulation)


        pygame.display.flip()

        #TEST
        #self.frame += 1
        #pygame.image.save(self.screen, "screenshots/screenshot" + str(self.frame) + ".jpeg")
        #/TEST

        self.clock.tick(60)

        keys = pygame.key.get_pressed()  # checking pressed keys
        return keys[pygame.K_SPACE], (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_DOWN]), keys[pygame.K_LEFT]

    def draw_level(self, copter_simulation):
        level = copter_simulation.level
        if self.who_to_follow == 'main' or self.who_to_follow >= len(copter_simulation.enemy_instances):
            cx = copter_simulation.copter.get_x()
        else:
            cx = copter_simulation.enemy_instances[self.who_to_follow].enemy.get_x()
        start = cx - self.get_view_offset(copter_simulation)
        end = start + self.size[0]
        start = min(len(level), max(0, start))
        end = min(len(level), max(0, end))
        start = int(start)
        end = int(end)
        ceiling_region = copter_simulation.level.ceiling[start:end]
        ground_region = copter_simulation.level.ground[start:end]
        ceiling_coords = [(x, y) for x, y in enumerate(list(ceiling_region))]
        # ceiling_pointlist = [(this, next) for this, next in izip(ceiling_coords, ceiling_coords[1:])]
        ground_coords = [(x, y) for x, y in enumerate(list(ground_region))]
        # ground_pointlist = [(this, next) for this, next in izip(ground_coords, ground_coords[1:])]
        c_cave = (150,160,170)
        # for point in ceiling_pointlist:
        #     pygame.draw.lines(self.screen, (255, color, 0), False, point, 2)
        # for point in ground_pointlist:
        #     pygame.draw.lines(self.screen, (255, color, 0), False, point, 2)
        pygame.draw.polygon(self.screen, c_cave, [[0,0]]+[[0,self.size[1]]] + ceiling_coords + [[self.size[0],self.size[1]]]+[[self.size[0],0]])
        pygame.draw.polygon(self.screen, c_cave, [[0,self.size[1]]] + [[0,self.size[1]]] + ground_coords + [[self.size[0],self.size[1]]])

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
        for particle in copter_simulation.smoke.frozen_particles:
            self.draw_smoke_particle(particle, copter_simulation)

    def draw_enemy_smoke(self, copter_simulation, index):
        smoke = copter_simulation.enemy_instances[index].smoke
        for particle in smoke.particles:
            self.draw_smoke_particle(particle, copter_simulation)
        for particle in smoke.frozen_particles:
            self.draw_smoke_particle(particle, copter_simulation)

    def np_to_screen_coord(self, np_vector, copter_simulation):
        if self.who_to_follow == 'main' or self.who_to_follow >= len(copter_simulation.enemy_instances): # follow main robot
            x = np_vector[0] - copter_simulation.copter.get_x()
            y = np_vector[1]
            view_offset = self.view_offset
        else:
            x = np_vector[0] - copter_simulation.enemy_instances[self.who_to_follow].enemy.get_x()
            y = np_vector[1]
            view_offset = self.enemy_view_offset
        return (view_offset + x, y)

    def get_view_offset(self, copter_simulation):
        return self.view_offset if self.who_to_follow == 'main' or self.who_to_follow >= len(copter_simulation.enemy_instances) else self.enemy_view_offset

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

    def draw_enemies_radars(self, copter_simulation):
        for ei in copter_simulation.get_living_enemy_instances():
            radar_system = copter_simulation.enemys_radar_system
            for radar in radar_system.radars:
                point, dist = radar.point, radar.dist
                if True:#(point,dist) == (None,None):
                    point,dist = radar.read(ei.enemy.position, copter_simulation.level)
                if True:#dist < 1:
                    dist = 0.5 + dist/2 # for drawing only
                    color = (255, int(150+(255-150)*dist), int(dist*255))
                    pygame.draw.circle(self.screen, color, self.np_to_screen_coord(point, copter_simulation), 5)

    def draw_object_radars(self, copter_simulation):
        if not copter_simulation.copter.exploded:
            self.draw_object_radar(copter_simulation.radar_system.enemy_radar, copter_simulation.copter.position, copter_simulation.get_living_enemy_instances(), (255,80,255), (255,255,255), copter_simulation)

    def draw_enemy_object_radars(self, copter_simulation):
        for i,ei in enumerate(copter_simulation.enemy_instances):
            enemy = ei.enemy
            if not enemy.exploded:
                self.draw_object_radar(copter_simulation.enemys_radar_system.shot_radar, enemy.position, copter_simulation.shots, (150,80,255), (255,255,255), copter_simulation)
                self.draw_object_radar(copter_simulation.enemys_radar_system.copter_radar, enemy.position,
                                       copter_simulation.get_living_copter_list(), (255, 80, 255), (255, 255, 255), copter_simulation)
                other_enemies = copter_simulation.get_living_enemy_instances(i)
                self.draw_object_radar(copter_simulation.enemys_radar_system.enemy_radar, enemy.position,
                                       other_enemies, (0, 255, 0), (255, 255, 255), copter_simulation)

    def draw_object_radar(self, radar, position, object_list, close_color, far_color, copter_simulation):
        dist_vec = radar.read_dist_vector(position, object_list, copter_simulation.level)
        c_close = np.array(close_color)
        c_far = np.array(far_color)
        c_diff = c_far - c_close
        for i in range(radar.number_of_neurons):
            if dist_vec[i] < 1.0:
                if radar.only_left_half:
                    angle = -(i + 0.5) * radar.angle_slice - np.pi / 2.0
                else:
                    angle = (i + 0.5) * radar.angle_slice - np.pi
                color = c_close + c_diff * dist_vec[i]
                self.draw_circle_at_angle(color, position, dist_vec[i] * 150, angle,
                                          copter_simulation)

    def draw_player_box(self, rect, copter_simulation):
        line_fraction = 0.4 # 1.0 = full bounding box ~0 = only a pixel at each corner
        margin = 4
        width = 3
        copter = copter_simulation.copter
        x = np.asscalar(rect.get_position()[0]) -1#-copter.width/4.0
        y = np.asscalar(rect.get_position()[1]) #-copter.width/4.0
        pos = np.array([[x], [y]])
        color = (255, 200, 0)
        radius = rect.width / 2.0 + margin
        line_length = line_fraction * radius
        cornerNW = np.array([[np.asscalar(pos[0]) - radius], [np.asscalar(pos[1]) - radius]])
        cornerNE = np.array([[np.asscalar(pos[0]) + radius], [np.asscalar(pos[1]) - radius]])
        cornerSE = np.array([[np.asscalar(pos[0]) + radius], [np.asscalar(pos[1]) + radius]])
        cornerSW = np.array([[np.asscalar(pos[0]) - radius], [np.asscalar(pos[1]) + radius]])
        lineRight =np.array([[line_length], [0.0]])
        lineDown = np.array([[0.0], [line_length]])
        lineLeft = np.array([[-line_length], [0.0]])
        lineUp =   np.array([[0.0], [-line_length]])
        #print self.np_to_screen_coord(cornerNW + lineDown, copter_simulation)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerNW + lineDown, copter_simulation), self.np_to_screen_coord(cornerNW, copter_simulation), width)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerNW + lineRight, copter_simulation), self.np_to_screen_coord(cornerNW, copter_simulation), width)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerSW + lineUp, copter_simulation), self.np_to_screen_coord(cornerSW, copter_simulation), width)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerSW + lineRight, copter_simulation), self.np_to_screen_coord(cornerSW, copter_simulation), width)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerNE + lineDown, copter_simulation), self.np_to_screen_coord(cornerNE, copter_simulation), width)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerNE + lineLeft, copter_simulation), self.np_to_screen_coord(cornerNE, copter_simulation), width)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerSE + lineLeft, copter_simulation), self.np_to_screen_coord(cornerSE, copter_simulation), width)
        pygame.draw.line(self.screen, color, self.np_to_screen_coord(cornerSE + lineUp, copter_simulation), self.np_to_screen_coord(cornerSE, copter_simulation), width)



    def unit_dir_vector(self, angle):
        return np.array([[np.cos(angle)],[-np.sin(angle)]])

    def draw_circle_at_angle(self, color, position, dist, angle, copter_simulation):
        p = position + dist * self.unit_dir_vector(angle)
        pygame.draw.circle(self.screen, color, self.np_to_screen_coord(p, copter_simulation), 5)

    def draw_shots(self, copter_simulation):
        for shot in copter_simulation.shots:
            x,y  = self.np_to_screen_coord(shot.position, copter_simulation)
            x = x - shot.width / 2.0
            y = y - shot.height/ 2.0
            pygame.draw.rect(self.screen, (255, 180, 0),
                             pygame.Rect(x, y, shot.width, shot.height))

    def draw_score(self, copter_simulation):
        score_color = (255, 255, 255)
        text_surface = self.score_font.render('Score: %d' % int(self.score), False, score_color)
        self.screen.blit(text_surface, (20, 20))

    def add_shot_score_text(self, hitPosition, copter_simulation):
        self.shot_score_texts.append(ScoreText(hitPosition))

    def update_shot_score_texts(self):
        for i,shot_score_text in enumerate(list(self.shot_score_texts)):
            if not shot_score_text.update():
                del self.shot_score_texts[i]


    def draw_shot_score_texts(self, copter_simulation):
        for shot_score_text in self.shot_score_texts:
            self.screen.blit(self.shot_score_surface, self.np_to_screen_coord(shot_score_text.position, copter_simulation))




