

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

