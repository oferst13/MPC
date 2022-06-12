import numpy as np
import cfg


class Tank:
    standard_size = 20
    all_tanks = []

    def __init__(self, name, n_tanks, init_storage, roof, dwellers
                 ):
        self.name = name
        self.n_tanks = n_tanks
        self.init_storage = init_storage
        self.roof = roof
        self.dwellers = dwellers
        self.tank_size = self.n_tanks * self.standard_size
        self.cur_storage = init_storage
        self.cur_sim_storage = init_storage
        self.overflows = np.zeros(cfg.sim_len)
        self.in_volume_forecast = None
        self.in_volume_actual = None
        self.daily_demands = None # currently with dt only
        Tank.all_tanks.append(self)

    def set_storage(self):
        pass

    def get_storage(self):
        pass

    def set_overflow(self):
        pass

    def get_overflow(self):
        pass

    def set_release(self):
        pass

    def tot_outflow(self):
        pass

    def set_rain_forecast(self, rain):
        self.in_volume_forecast = rain * self.roof / 1000

    def set_demands(self, demand_pattern):
        self.daily_demands = self.dw
