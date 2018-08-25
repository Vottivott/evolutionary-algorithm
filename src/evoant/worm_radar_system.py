import numpy as np

from ..radar import BinaryRadar

class WormRadarSystem:
    def __init__(self, num_segments):
        num_balls = num_segments + 1

        number_of_neurons = 8
        only_bottom_half = True
        self.ground_contact_radars = [BinaryRadar(number_of_neurons,  only_bottom_half) for _ in range(num_balls)]

        number_of_neurons = 16
        only_bottom_half = False
        self.muscle_direction_radars = [BinaryRadar(number_of_neurons, only_bottom_half) for _ in range(num_segments)]