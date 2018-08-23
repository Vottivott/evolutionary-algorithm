import numpy as np

class Circular:
    def __init__(self, position, radius, num_coll_points):
        self.position = position
        self.radius = radius
        self.radius_squared = radius * radius
        if num_coll_points:
            self.ceiling_coll_points, self.floor_coll_points = self.init_coll_points(num_coll_points)

    def get_position(self):
        return self.position

    def get_top(self):
        return self.position[1] - self.radius

    def get_bottom(self):
        return self.position[1] + self.radius

    def get_left(self):
        return self.position[0] - self.radius

    def get_right(self):
        return self.position[0] + self.radius

    def get_x(self):
        return self.position[0]

    def get_y(self):
        return self.position[1]

    def collides_with(self, circular):
        return np.dot(self.position, circular.position) <= self.radius_squared + 2*self.radius*circular.radius + circular.radius_squared

    def contains_point(self, p):
        return np.dot(self.position, p) <= self.radius_squared

    def init_coll_points(self, num_points):
        ceil_dirs = np.linspace(0, np.pi, num_points / 2)
        floor_dirs = np.linspace(-np.pi, 0, num_points / 2)
        ceil_points = [self.radius * np.array([[np.cos(d)], [-np.sin(d)]]) for d in ceil_dirs]
        floor_points = [self.radius * np.array([[np.cos(d)], [-np.sin(d)]]) for d in floor_dirs]
        return ceil_points, floor_points

