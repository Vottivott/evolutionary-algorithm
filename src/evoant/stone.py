from evoant.ball import Ball


class Stone(Ball):
    def __init__(self, position, radius, initial_strength, num_foods_inside):
        Ball.__init__(self, position, radius, 1.0, 1.0, 0.0,
                      10000.0)
        # self.position = position
        # self.radius = radius
        self.initial_strength = initial_strength
        self.strength = initial_strength
        self.num_foods_inside = num_foods_inside
        self.reset()

    def reset(self):
        self.strength = self.initial_strength

