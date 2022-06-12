import numpy as np
import cfg


class Pipe:
    all_pipes = []

    def __init__(self, name,  length, diameter, slope):
        self.name = name
        self.length = length
        self.diameter = diameter
        self.slope = slope
        self.alpha = (0.501 / cfg.manning) * (diameter ** (1 / 6)) * (slope ** 0.5)
        Pipe.all_pipes.append(self)

    def calc_q_outlet(self):
        pass

    def get_q_outlet(self):
        pass

