import numpy as np

class ObjectRadar:
    def __init__(self, number_of_neurons, x_step_size, max_num_steps, max_dist, only_left_half):
        self.number_of_neurons = number_of_neurons
        self.x_step_size = x_step_size
        self.max_num_steps = max_num_steps
        self.max_dist = max_dist
        self.only_left_half = only_left_half
        self.angle_slice = 2.0*np.pi / number_of_neurons if not only_left_half else np.pi / number_of_neurons # slice watched by a single neuron

    def read_dist_vector(self, position, object_list, level):
        dist_vector = np.ones((self.number_of_neurons,1))
        for object in object_list:
            p = object.get_position()
            diff = p - position
            if self.only_left_half and diff[0] > 0:
                continue
            dist = np.linalg.norm(diff)
            if dist > self.max_dist:
                continue
            num_steps = min(int(abs(diff[0]) / self.x_step_size), self.max_num_steps)
            step = 0 if num_steps == 0 else diff / num_steps
            mid_pos = np.copy(position)
            viewable = True
            for i in range(num_steps):
                mid_pos += step
                if level.collides_with_point(mid_pos):
                    viewable = False
                    break
            if viewable:
                dist_value = dist / self.max_dist
                if self.only_left_half:
                    angle = np.pi/2.0 + np.arctan2(-diff[1], -diff[0])
                    dir_index = int(angle / self.angle_slice)
                else:
                    angle = np.pi + np.arctan2(-diff[1],diff[0])
                    dir_index = int(angle / self.angle_slice)
                dist_vector[dir_index] = min(dist_vector[dir_index], dist_value)
        return dist_vector










class Radar:
    def __init__(self, direction, max_steps, x_step_size):
        # self.direction = direction
        direction_vector = np.array([[np.cos(direction)],[-np.sin(direction)]])
        self.step = np.array([[x_step_size], [x_step_size * direction_vector[1]/direction_vector[0]]])
        self.single_pixel_step = self.step / abs(x_step_size)
        self.single_pixel_fraction = 1.0 / abs(x_step_size)
        self.x_step_size = abs(x_step_size)
        self.max_steps = max_steps
        self.step_dist = np.linalg.norm(self.step)
        self.point = None
        self.dist = None

    def read(self, position, level):
        p = np.copy(position)
        for n in range(1,self.max_steps+1):
            p += self.step
            if level.collides_with_point(p):
                for i in range(self.x_step_size-1):
                    if not level.collides_with_point(p-self.single_pixel_step):
                        break
                    else:
                        p -= self.single_pixel_step
                        n -= self.single_pixel_fraction
                return (p,n/float(self.max_steps))
        self.point, self.dist = p,self.max_steps/float(self.max_steps)

        return (self.point, self.dist)

    # def read_rect(self, position, rect_list):
    #     p = np.copy(position)
    #     for n in range(1,self.max_steps+1):
    #         p += self.step
    #         for rect in rect_list:
    #             if rect.contains_point(p):
    #                 for i in range(self.x_step_size-1):
    #                     if not rect.contains_point(p - self.single_pixel_step):
    #                         break
    #                     else:
    #                         p -= self.single_pixel_step
    #                         n -= self.single_pixel_fraction
    #                 return (p,n/float(self.max_steps))
    #     self.point, self.dist = p,self.max_steps/float(self.max_steps)
    #     print (self.point, self.dist)
    #     return (self.point, self.dist)