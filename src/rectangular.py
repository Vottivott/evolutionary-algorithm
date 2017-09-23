

class Rectangular:
    def __init__(self, position, width, height):
        self.position = position
        self.width = width
        self.height = height

    def get_top(self):
        return self.position[1] - self.height/2.0

    def get_bottom(self):
        return self.position[1] + self.height/2.0

    def get_left(self):
        return self.position[0] - self.width/2.0

    def get_right(self):
        return self.position[0] + self.width/2.0

    def get_x(self):
        return self.position[0]

    def get_y(self):
        return self.position[1]

    def collides_with(self, rect):
        return self.get_right() > rect.get_left() and self.get_left() < rect.get_right() \
           and self.get_bottom() > rect.get_top() and self.get_top() < rect.get_bottom()

    def contains_point(self, p):
        return p[0] > self.get_left() and p[0] < self.get_right() and p[1] > self.get_top() and p[1] < self.get_bottom()



