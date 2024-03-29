import copy
from scipy import integrate
import numpy as np
import cfg


class Node:
    all_nodes = []
    lat_nodes = []

    def __init__(self, name, receiving_from=[], giving_to=[], tank_node=False, lat_node=False):
        self.name = name
        self.receiving_from = receiving_from
        self.giving_to = giving_to
        self.tank_node = tank_node
        self.lat_node = lat_node
        self.lat_flows = None
        Node.all_nodes.append(self)
        if self.lat_node:
            Node.lat_nodes.append(self)

    @classmethod
    def set_lat_flows_all(cls, swmm_lat_flows):
        for num, node in enumerate(cls.lat_nodes):
            node.set_lat_flows(swmm_lat_flows[num, :] / 1000)

    @classmethod
    def reset_lat_flows(cls):
        for node in cls.lat_nodes:
            node.lat_flows = None

    def handle_flow(self, timestep, swmm):
        inflow: float = 0
        if self.tank_node:
            for tank in self.receiving_from:
                inflow += (tank.overflows[timestep] + tank.release_volume[timestep]) / cfg.dt
        else:
            for pipe in self.receiving_from:
                inflow += pipe.outlet_Q[timestep]
            if self.lat_node and swmm:
                try:
                    inflow += self.lat_flows[timestep]
                except IndexError:
                    inflow += 0.0

        for pipe in self.giving_to:
            pipe.inlet_Q[timestep] = inflow / len(self.giving_to)

    def get_zero_Q(self):
        last_Q_list = []
        for pipe in self.receiving_from:
            if np.sum(np.nonzero(pipe.outlet_Q)) > 0:
                last_Q_list.append(np.max(np.nonzero(pipe.outlet_Q)))
            else:
                last_Q_list.append(0)
        if self.lat_node and (self.lat_flows is not None and np.sum(self.lat_flows) > 0.1):
            last_Q_list.append(np.max(np.nonzero(self.lat_flows > 0.00001)))
        return max(last_Q_list) + 1

    def get_max_Q(self):
        inflows = np.zeros((len(self.receiving_from), len(self.receiving_from[0].outlet_Q)))
        for idx, pipe in enumerate(self.receiving_from):
            inflows[idx, :] = pipe.outlet_Q
        if self.lat_node:
            lateral = np.pad(self.lat_flows, (0, len(self.receiving_from[0].outlet_Q) - len(self.lat_flows)), 'constant')
            inflows = np.vstack((inflows, lateral))
        summed = np.sum(inflows, axis=0)
        return np.max(summed)

    def get_outflow_volume(self):
        zero_Q = self.get_zero_Q()
        tot_volume = 0.0
        for pipe in self.receiving_from:
            tot_volume += integrate.simps(pipe.outlet_Q[:zero_Q], cfg.t[:zero_Q]) * cfg.dt
        if self.lat_node is True:
            tot_volume += integrate.simps(self.lat_flows, cfg.t[:len(self.lat_flows)]) * cfg.dt
        return tot_volume

    def get_flow(self, timestep):
        inflow: float = 0
        for pipe in self.receiving_from:
            inflow += pipe.outlet_Q[timestep]
            if self.lat_node:
                inflow += self.lat_flows[timestep]
        return inflow

    def set_lat_flows(self, flows):
        self.lat_flows = flows
