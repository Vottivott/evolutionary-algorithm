class Stone:
    def __init__(self, position, radius, initial_strength, num_foods_inside):
        self.position = position
        self.radius = radius
        self.initial_strength = initial_strength
        self.num_foods_inside = num_foods_inside
        self.reset()

    def reset(self):
        self.strength = self.initial_strength

