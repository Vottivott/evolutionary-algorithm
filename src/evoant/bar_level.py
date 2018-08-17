from itertools import izip

import numpy as np

from evoant.stone import Stone


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

def generate_bar_level_with_stones(length, close_end=True):
    lvl = generate_bar_level(length, close_end)
    lvl.stones = []
    stone_density_per_1000 = 2.0
    num_stones = int(stone_density_per_1000 * length / 1000.0)
    for i in range(num_stones):
        pos = np.clip(length * np.random.rand(), 0.0, length - 3)
        ceil = lvl.get_ceiling(pos)
        ground = lvl.get_ground(pos)
        half = (ground - ceil) / 2.0
        stone = get_random_stone(np.array([[pos],[0.0]]))
        is_ceil = float(np.random.rand()) > 0.5
        if is_ceil:
            stone.position[1] = ceil + min(0.0, half - stone.radius)
            stone.position[1] -= np.random.rand() * stone.radius / 2.0
        else:
            stone.position[1] = ground - min(0.0, half - stone.radius)
            stone.position[1] += np.random.rand() * stone.radius / 2.0
        lvl.stones.append(stone)
    # for stone in lvl.stones:
    #     print stone.radius
    lvl.stones = sorted(lvl.stones, key = lambda stone: stone.position[0])
    return lvl

def generate_bar_level(length, close_end=True):
    n = length / 70.0
    basic_shape = _get_randcurve(length, n, 40)
    ceiling = basic_shape + _get_randcurve(length, 1.5*n, 10) + _get_randcurve(length, 1.7*n, 4)
    ceiling = 125 + _smoothify(ceiling)
    ground = basic_shape + _get_randcurve(length, 1.5*n, 18) + _get_randcurve(length, 1.7*n, 8)
    ground = 305 + _smoothify(ground) #prev 305
    if close_end:
        ceiling[-1] = ground[-1]
        ceiling[0] = ground[0]


    bar_width = 50
    return BarLevel(ceiling[::bar_width], ground[::bar_width], float(bar_width))


def generate_planar_bar_level(length, close_end=True):
    n = length / 70.0
    amp_factor = 0.01
    basic_shape = _get_randcurve(length, n, 10)
    ceiling = basic_shape + _get_randcurve(length, 1.5*n, amp_factor * 10) + _get_randcurve(length, 1.7*n, amp_factor * 4)
    ceiling = 125 + _smoothify(ceiling)
    ground = basic_shape + _get_randcurve(length, 1.5*n, amp_factor * 18) + _get_randcurve(length, 1.7*n, amp_factor * 8)
    ground = 305 + _smoothify(ground) #prev 305
    if close_end:
        ceiling[-1] = ground[-1]
        ceiling[0] = ground[0]


    bar_width = 50
    return BarLevel(ceiling[::bar_width], ground[::bar_width], float(bar_width))

def get_random_stone(position):
    radius = float(np.clip(np.random.normal(50.0, 20.0), 20.0, 100.0))
    strength = np.clip(np.random.normal(1.0, 0.2), 0.5, 1.5) * radius
    # if np.random.rand() > 0.2:
    num_foods_inside = np.clip(np.random.normal(7.0, 5.0), 1.0, 1000.0)
    if np.random.rand() > radius / 150.0:
        num_foods_inside += np.clip(np.random.normal(3.0 * radius/100.0, 1.0), 0.0, 15.0)
    # else:
    #     num_foods_inside = 0.0
    return Stone(position, radius, strength, num_foods_inside)


class BarLevel:
    def __init__(self, ceiling, ground, bar_width):
        self.ceiling = ceiling
        self.ground = ground
        self.bar_width = bar_width


    def get_ceiling(self, x):
        index = x/self.bar_width
        left_index = min(len(self.ceiling)-2, int(index))
        frac = index - left_index
        left = self.ceiling[left_index]
        right = self.ceiling[left_index+1]
        diff = right - left
        return left + diff * frac

    def get_ground(self, x):
        index = x/self.bar_width
        left_index = min(len(self.ground) - 2, int(index))
        frac = index - left_index
        left = self.ground[left_index]
        right = self.ground[left_index+1]
        diff = right - left
        return left + diff * frac

    def collides_with_rectangular(self, rectangular):
        # Simplified collision using only the top and bottom center points of the rectangle
        if self.ceiling_collides_with_rectangular(rectangular):
            return True
        if self.ground_collides_with_rectangular(rectangular):
            return True
        return False

    def collides_with_multipoint(self, rectangular):
        # Simplified collision using only the top and bottom center points of the rectangle,
        # but using the left corner, center and right corner points to make it more robust
        if self.ceiling_collides_with_multipoint(rectangular):
            # print "ceiling"
            return True
        if self.ground_collides_with_multipoint(rectangular):
            # print "ground"
            return True
        return False

    def collides_with_point(self, point):
        x = int(point[0])
        return x < 0 or x >= len(self.ceiling) or \
               (point[1] < self.get_ceiling(x) or point[1] > self.get_ground(x))


    def y_center(self, x):
        return (self.get_ceiling(x) + self.get_ground(x)) / 2.0

    def bounce_direction(self, direction, rectangular):
        if self.ceiling_collides_with_rectangular(rectangular):
            slope = self.calculate_slope_rectangular(rectangular, self.ceiling)
            return self.bounce(direction, slope)
        if self.ground_collides_with_rectangular(rectangular):
            slope = self.calculate_slope_rectangular(rectangular, self.ground)
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

    def ceiling_collides_with_rectangular(self, rectangular):
        x = int(rectangular.get_x())
        return x < 0 or x >= len(self.ceiling) or self.get_ceiling(x) > rectangular.get_top()

    def ground_collides_with_rectangular(self, rectangular):
        x = int(rectangular.get_x())
        return x >= len(self.ground) or self.get_ground(x) < rectangular.get_bottom()

    def ceiling_collides_with_multipoint(self, rectangular):
        x_list = [int(rectangular.get_x()), int(rectangular.get_left()), int(rectangular.get_right())]
        for x in x_list:
            if x < 0 or x >= len(self.ceiling) or self.get_ceiling(x) > rectangular.get_top():
                return True
        return False

    def ground_collides_with_multipoint(self, rectangular):
        x_list = [int(rectangular.get_x()), int(rectangular.get_left()), int(rectangular.get_right())]
        for x in x_list:
            if x < 0 or x >= len(self.ground) or self.get_ground(x) < rectangular.get_bottom():
                return True
        return False

    def calculate_slope_rectangular(self, rectangular, curve):
        x = rectangular.get_x()
        lx = int(x)
        rx = lx + 1
        if rx >= len(curve):
            return 100000.0
        return curve[rx] - curve[lx]

    # CIRCULAR

    def collides_with_circular(self, circular):
        for cp in circular.ceiling_coll_points:
            p = circular.get_position() + cp
            if self.collides_with_point(p):
                return True
        for cp in circular.floor_coll_points:
            p = circular.get_position() + cp
            if self.collides_with_point(p):
                return True
        return False









    def get_ceiling_collision_circular(self, circular):
    # Return the median of the hit points
        coll = []
        for cp in circular.ceiling_coll_points:
            p = circular.get_position() + cp
            if self.collides_with_point(p):
                coll.append(p)
        if len(coll) == 0:
            return None
        return coll[len(coll)/2]

    def get_ground_collision_circular(self, circular):
    # Return the median of the hit points
        coll = []
        for cp in circular.floor_coll_points:
            p = circular.get_position() + cp
            if self.collides_with_point(p):
                coll.append(p)
        if len(coll) == 0:
            return None
        return coll[len(coll)/2]

    def calculate_slope_x(self, x, curve):
        lx = int(x)
        rx = lx + 1
        if rx >= len(curve):
            return 100000.0
        return curve[rx] - curve[lx]

    def calculate_slope_x_smoother(self, x, curve):
        lx = int(x)
        rx = lx + 1
        llx = lx-1
        rrx = rx#+1
        if rrx >= len(curve) or llx < 0:
            return 100000.0
        return (curve[rrx] - curve[llx]) / 2.0


    def get_normal(self, slope):
        n = np.array([[1.0], [-1.0 / slope]])
        return n / (np.dot(n.T, n) ** 0.5)

    # def collides_with_circular(self, circular):
    #     if self.ceiling_collides_with_circular(circular):
    #         return True
    #     if self.ground_collides_with_circular(circular):
    #         return True
    #     return False

    def normal_direction_circular(self, circular):
        ground_coll = self.get_ground_collision_circular(circular)
        if ground_coll is not None and ground_coll[0] >= 0 and ground_coll[0] < len(self.ground):
            slope = self.calculate_slope_x(ground_coll[0], self.ground)
            return self.get_normal(slope)
            #return self.bounce(direction, slope)
        ceil_coll = self.get_ceiling_collision_circular(circular)
        if ceil_coll is not None and ceil_coll[0] >= 0 and ceil_coll[0] < len(self.ceiling):
            slope = self.calculate_slope_x(ceil_coll[0], self.ceiling)
            return self.get_normal(slope)
            #return self.bounce(direction, slope)
        return None


    # def overlap_and_normal_direction_circular(self, circular):
    #     ground_coll = self.get_ground_collision_circular(circular)
    #     if ground_coll is not None and ground_coll[0] >= 0 and ground_coll[0] < len(self.ground):
    #         slope = self.calculate_slope_x(ground_coll[0], self.ground)
    #         normal = self.get_normal(slope)
    #         overlap = ground_coll *
    #         return overlap, normal
    #         #return self.bounce(direction, slope)
    #     ceil_coll = self.get_ceiling_collision_circular(circular)
    #     if ceil_coll is not None and ceil_coll[0] >= 0 and ceil_coll[0] < len(self.ceiling):
    #         slope = self.calculate_slope_x(ceil_coll[0], self.ceiling)
    #         normal = self.get_normal(slope)
    #         return overlap, normal
    #         #return self.bounce(direction, slope)
    #     return None, None


