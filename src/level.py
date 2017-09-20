from itertools import izip

import numpy as np

def _get_randcurve(length, n, amplitude):
    x = np.linspace(0, 2*np.pi, n)
    xvals = np.linspace(0, 2*np.pi, length)
    curve = np.random.normal(0, 1, n)
    yinterp = np.interp(xvals, x, curve)
    curve = yinterp*amplitude
    return curve

def _smoothify(curve):
    smooth_factor = 20 #120  # 240#120
    left, right = -smooth_factor, 0
    running_sum = curve[left]
    n = len(curve)
    smooth_curve = []
    for left, right in zip([0] * (smooth_factor / 2) + range(0, n - smooth_factor / 2),
                           range(smooth_factor / 2, n) + [n - 1] * (smooth_factor / 2)):
        running_sum = running_sum - curve[left] + curve[right]
        smooth_curve.append(running_sum * 1.0 / smooth_factor)
    return np.array(smooth_curve)

def generate_level(length):
    n = length / 70.0
    basic_shape = _get_randcurve(length, n, 40)
    ceiling = basic_shape + _get_randcurve(length, 1.5*n, 10) + _get_randcurve(length, 1.7*n, 4)
    ceiling = 125 + _smoothify(ceiling)
    ground = basic_shape + _get_randcurve(length, 1.5*n, 18) + _get_randcurve(length, 1.7*n, 8)
    ground = 305 + _smoothify(ground)

    return Level(ceiling, ground)





class Level:
    def __init__(self, ceiling, ground):
        self.ceiling = ceiling
        self.ground = ground

    def collides_with(self, rectangular):
        # Simplified collision using only the top and bottom center points of the rectangle
        if self.ceiling_collides_with(rectangular):
            return True
        if self.ground_collides_with(rectangular):
            return True

    def collides_with_point(self, point):
        return point[1] < self.ceiling[int(point[0])] or point[1] > self.ground[int(point[0])]


    def y_center(self, x):
        return (self.ceiling[x] + self.ground[x]) / 2.0

    def bounce_direction(self, direction, rectangular):
        if self.ceiling_collides_with(rectangular):
            slope = self.calculate_slope(rectangular, self.ceiling)
            return self.bounce(direction, slope)
        if self.ground_collides_with(rectangular):
            slope = self.calculate_slope(rectangular, self.ground)
            return self.bounce(direction, slope)
        return None

    def bounce(self,direction, slope):
        normal = np.array([[1.0], [-1.0 / slope]])
        projection = (direction.T.dot(normal) / direction.T.dot(direction)) * normal
        bounce_dir = (projection - direction)
        bounce_dir *= 1.0 / (bounce_dir.T.dot(bounce_dir)) ** 0.5
        return bounce_dir

    def __len__(self):
        return len(self.ceiling)

    def ceiling_collides_with(self, rectangular):
        return self.ceiling[int(rectangular.get_x())] > rectangular.get_top()

    def ground_collides_with(self, rectangular):
        return self.ground[int(rectangular.get_x())] < rectangular.get_bottom()

    def calculate_slope(self, rectangular, curve):
        x = rectangular.get_x()
        lx = int(x)
        rx = lx + 1
        return curve[rx] - curve[lx]


