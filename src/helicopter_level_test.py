import time
from itertools import *
import pygame, sys
from pygame.locals import *
import numpy as np
from scipy.interpolate import UnivariateSpline

pygame.init()
size = w,h = 720, 720
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()


#curve = np.repeat(curve,20, axis=0)
#curve = np.interp(xvals, x, curve)
def get_randcurve(n, amplitude):
    x = np.linspace(0, 2*np.pi, n)
    xvals = np.linspace(0, 2*np.pi, 560)
    curve = np.random.normal(0, 1, n)
    yinterp = np.interp(xvals, x, curve)
    curve = yinterp*amplitude
    return curve

def get_curve():
    curve = get_randcurve(10, 120)# + get_randcurve(20, 30) + get_randcurve(40, 15) + get_randcurve(80, 8)

    curve *= .2

    # smooth_factor = 20#120#240#120
    # left, right = -smooth_factor, 0
    # running_sum = curve[left]
    # n = len(curve)
    # smooth_curve = []
    # for left, right in zip([0]*(smooth_factor/2) + range(0,n-smooth_factor/2), range(smooth_factor/2, n) + [n-1]*(smooth_factor/2)):
    #     running_sum = running_sum - curve[left] + curve[right]
    #     smooth_curve.append(running_sum*1.0/smooth_factor)
    smooth_curve = curve

    x_offset, y_offset = 200, 60
    coords = [(y + y_offset, x + x_offset) for y, x in enumerate(list(smooth_curve))]
    pointlists = [(this, next) for this, next in izip(coords, coords[1:])]
    return pointlists

pointlists = get_curve()
color = min(np.random.normal(200,30,1), 255)
trace = False

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and event.key == K_t:
            trace = not trace

    #screen.fill((255,255,255))
    #screen.fill((255, 255, 255))
    s = pygame.Surface(size)  # the size of your rect
    if trace:
        s.set_alpha(28)  # alpha level
    s.fill((0, 0, 0))  # this fills the entire surface
    screen.blit(s, (0, 0))  # (0,0) are the top-left coordinates

    keys = pygame.key.get_pressed()  # checking pressed keys
    if keys[pygame.K_SPACE]:
        pointlists = get_curve()
        color = min(np.random.normal(200,30,1), 255)

    for pointlist in pointlists:
        pygame.draw.lines(screen, (255, color, 0), False, pointlist, 2)
        pygame.draw.lines(screen, (255, color, 0), False, [(x,y+100) for (x,y) in pointlist], 2)

    pygame.display.flip()
    clock.tick(24)