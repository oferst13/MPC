import copy
from scipy import integrate
import numpy as np
import cfg


class Node:
    all_nodes = []

    def __init__(self, name, receiving_from=[], giving_to=[], tank_node=False, lat_node=False):
        self.name = name
        self.receiving_from = receiving_from
        self.giving_to = giving_to
        self.tank_node = tank_node
        self.lat_node = lat_node
        self.lat_flows = None
        Node.all_nodes.append(self)

    def handle_flow(self, timestep, swmm=False):
        inflow: float = 0
        if self.tank_node:
            for tank in self.receiving_from:
                inflow += (tank.overflows[timestep] + tank.release_volume[timestep]) / cfg.dt
        else:
            for pipe in self.receiving_from:
                inflow += pipe.outlet_Q[timestep]
        for pipe in self.giving_to:
            pipe.inlet_Q[timestep] = inflow / len(self.giving_to)

    def get_zero_Q(self):
        last_Q_list = []
        for pipe in self.receiving_from:
            if np.sum(np.nonzero(pipe.outlet_Q)) > 0:
                last_Q_list.append(np.max(np.nonzero(pipe.outlet_Q)))
            else:
                last_Q_list.append(0)
        return max(last_Q_list) + 1

    def get_max_Q(self): # this works only with 1 pipe for now
        max_Q = 0.0
        for pipe in self.receiving_from:
            max_Q += np.max(pipe.outlet_Q)
        return max_Q

    def get_outflow_volume(self):
        zero_Q = self.get_zero_Q()
        tot_volume = 0.0
        for pipe in self.receiving_from:
            tot_volume += integrate.simps(pipe.outlet_Q[:zero_Q], cfg.t[:zero_Q]) * cfg.dt
        return tot_volume

    def get_flow(self, timestep):
        inflow: float = 0
        for pipe in self.receiving_from:
            inflow += pipe.outlet_Q[timestep]
        return inflow

    def set_lat_flows(self, timestep, flows):
        self.lat_flows = flows
