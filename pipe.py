import copy

import numpy as np
import cfg


class Pipe:
    all_pipes = []

    def __init__(self, name, length, diameter, slope):
        self.name = name
        self.length = length
        self.diameter = diameter
        self.slope = slope
        self.alpha = (0.501 / cfg.manning) * (diameter ** (1 / 6)) * (slope ** 0.5)
        self.inlet_Q = None
        self.outlet_Q = None
        self.first_Q = None
        self.reset_pipe(cfg.forecast_len)
        Pipe.all_pipes.append(self)

    def calc_q_outlet(self, timestep):
        inlet_A = (self.inlet_Q[timestep] / self.alpha) ** (1 / cfg.beta)
        last_outlet_A = (self.outlet_Q[timestep - 1] / self.alpha) ** (1 / cfg.beta)
        constant = self.alpha * cfg.beta * (cfg.dt / self.length)
        out_A = last_outlet_A - constant * (((inlet_A + last_outlet_A) / 2) ** (cfg.beta - 1)) * (last_outlet_A - inlet_A)
        self.outlet_Q[timestep] = self.alpha * (out_A ** cfg.beta)

    @classmethod
    def get_tot_Q(cls, timestep):
        tot_Q: float = 0
        for pipe in cls.all_pipes:
            tot_Q += pipe.inlet_Q[timestep] + pipe.outlet_Q[timestep]
        return tot_Q

    def reset_pipe(self, duration):
        self.inlet_Q = np.zeros(duration, dtype=np.longfloat)
        self.outlet_Q = np.zeros(duration, dtype=np.longfloat)
