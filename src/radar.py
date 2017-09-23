import numpy as np

class Radar:
    def __init__(self, direction, max_steps, x_step_size):
        # self.direction = direction
        direction_vector = np.array([[np.cos(direction)],[-np.sin(direction)]])
        self.step = np.array([[x_step_size], [x_step_size * direction_vector[1]/direction_vector[0]]])
        self.max_steps = max_steps
        self.step_dist = np.linalg.norm(self.step)
        self.point = None
        self.dist = None

    def read(self, position, level):
        p = np.copy(position)
        for n in range(1,self.max_steps+1):
            p += self.step
            if level.collides_with_point(p):
                return (p,n/float(self.max_steps))
        self.point, self.dist = p,self.max_steps/float(self.max_steps)
        return (self.point, self.dist)

    def read_rect(self, position, rect_list):
        p = np.copy(position)
        for n in range(1,self.max_steps+1):
            p += self.step
            for rect in rect_list:
                if rect.contains_point(p):
                    return (p,n/float(self.max_steps))
        self.point, self.dist = p,self.max_steps/float(self.max_steps)
        return (self.point, self.dist)