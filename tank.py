import copy

import numpy as np
import cfg


class Tank:
    standard_size = 10
    standard_orifice = 0.05
    standard_diameter = 2.55
    all_tanks = []

    # def __init__(self, name, n_tanks, init_storage, roof, dwellers):
    def __init__(self, dictionary):
        # self.name = name
        # self.n_tanks = n_tanks
        # self.init_storage = init_storage
        # self.roof = roof
        # self.dwellers = dwellers
        for att, val in dictionary.items():
            setattr(self, att, val)

        self.tank_size = self.n_tanks * self.standard_size
        self.cur_storage = None
        self.orifice_A = ((self.standard_orifice / 2) ** 2) * np.pi
        self.footprint = ((self.standard_diameter / 2) ** 2) * np.pi
        self.overflows = None
        self.overflows_act = None
        self.releases = None
        self.releases_act = None
        self.release_volume = None
        self.release_volume_act = None
        self.rw_supply = None
        self.rw_supply_act = None
        self.all_storage = None
        self.all_storage_act = None
        self.inflow_forecast = None
        self.inflow_actual = None
        self.daily_demands = None  # currently with dt only
        self.next_init_storage = self.init_storage
        self.reset_tank(cfg.forecast_len, 'factory')
        Tank.all_tanks.append(self)

    @classmethod
    def get_tot_storage(cls):
        tot_storage: float = 0
        for tank in cls.all_tanks:
            tot_storage += tank.cur_storage
        return tot_storage

    @classmethod
    def get_tot_outflow(cls, timestep):
        tot_outflow: float = 0
        for tank in cls.all_tanks:
            tot_outflow += (tank.overflows[timestep] + tank.release_volume[timestep]) / cfg.dt
        return tot_outflow

    @classmethod
    def get_cum_overflow(cls):
        cum_overflow = np.zeros(cfg.sim_len)
        for tank in cls.all_tanks:
            cum_overflow += tank.overflows
        return np.sum(cum_overflow)

    @classmethod
    def get_cum_release(cls):
        cum_release = np.zeros(cfg.sim_len)
        for tank in cls.all_tanks:
            cum_release += tank.release_volume
        return np.sum(cum_release)

    @classmethod
    def get_cum_outflow(cls):
        return cls.get_cum_release() + cls.get_cum_overflow()

    @classmethod
    def get_last_overflow(cls):
        last_overflow_list = []
        for tank in cls.all_tanks:
            if np.sum(tank.overflows) > 0:
                last_overflow_list.append(np.max(np.nonzero(tank.overflows)))
        if last_overflow_list:
            return max(last_overflow_list)
        else:
            return 0

    @classmethod
    def set_inflow_forecast_all(cls, forecast_rain):
        for tank in cls.all_tanks:
            tank.set_inflow_forecast(forecast_rain)

    @classmethod
    def set_daily_demands_all(cls, demands_pd):
        for tank in cls.all_tanks:
            tank.set_daily_demands(demands_pd)

    @classmethod
    def set_next_cycle(cls):
        for tank in cls.all_tanks:
            tank.next_init_storage = tank.cur_storage
            tank.reset_tank(cfg.forecast_len)
            tank.cur_storage = tank.next_init_storage

    @classmethod
    def set_releases_all(cls, rel_array):
        for num, tank in enumerate(Tank.all_tanks):
            tank.set_releases(rel_array[num, :])

    @classmethod
    def reset_all(cls, duration, reset_type):
        for tank in Tank.all_tanks:
            tank.reset_tank(duration, reset_type)

    @classmethod
    def get_rw_supply_all(cls):
        tot_supply = 0.0
        for tank in Tank.all_tanks:
            tot_supply += tank.get_rw_supply()
        return tot_supply

    def calc_release(self, timestep, last_overflow):
        if timestep <= last_overflow:
            release_deg = self.releases[timestep // int(cfg.release_dt / cfg.dt)]
        else:
            release_deg = 0.0
        release_Q = self.n_tanks * self.orifice_A * cfg.Cd \
                    * np.sqrt(2 * 9.81 * (self.cur_storage / (self.n_tanks * self.footprint))) * 0.1 * release_deg
        release_vol = release_Q * cfg.dt
        self.cur_storage -= release_vol
        self.release_volume[timestep] = copy.copy(release_vol)
        if self.cur_storage < 0.0:
            self.cur_storage += release_vol
            self.release_volume[timestep] = copy.copy(self.cur_storage)
            self.cur_storage = 0.0

        '''
        for tank in cls.all_tanks:
            tank.next_init_storage = tank.cur_storage
            tank.reset_tank(cfg.forecast_len)
            tank.cur_storage = tank.next_init_storage
        '''

    def reset_tank(self, duration, reset_type):
        reset_types = ['factory', 'cycle', 'iter']
        if reset_type not in reset_types:
            raise ValueError('Invalid reset type. Expected on of: %s' % reset_types)
        self.overflows = np.zeros(duration)
        self.releases = np.zeros(int(duration / (cfg.release_dt / cfg.dt)))
        self.release_volume = np.zeros(duration)
        self.rw_supply = np.zeros(duration)
        self.all_storage = np.zeros(duration)
        if reset_type == 'factory':
            self.cur_storage = self.init_storage
            self.all_storage[0] = self.init_storage
        elif reset_type == 'cycle':
            self.next_init_storage = self.cur_storage
            self.all_storage[0] = self.next_init_storage
        elif reset_type == 'iter':
            self.cur_storage = self.next_init_storage
            self.all_storage[0] = self.next_init_storage

    def set_inflow_forecast(self, rain):
        self.inflow_forecast = rain * self.roof / 1000

    def set_daily_demands(self, demand_pattern):
        self.daily_demands = demand_pattern * self.dwellers / 1000

    def tank_fill(self, timestep):
        cur_rain_volume = self.inflow_forecast[int(timestep // (cfg.rain_dt / cfg.dt))] * (cfg.dt / cfg.rain_dt)
        self.cur_storage += cur_rain_volume
        if self.cur_storage > self.tank_size:
            overflow = self.cur_storage - self.tank_size
            self.cur_storage = self.tank_size
            self.overflows[timestep] = overflow

    def set_releases(self, release_vec):
        self.releases = release_vec

    def rw_use(self, timestep):
        demand = self.daily_demands[timestep % self.daily_demands.shape[0]]
        self.cur_storage -= demand
        self.rw_supply[timestep] = copy.copy(demand)
        if self.cur_storage < 0:
            self.cur_storage += demand
            self.rw_supply[timestep] = copy.copy(self.cur_storage)
            self.cur_storage = 0
        self.all_storage[timestep] = copy.copy(self.cur_storage)

    def get_rw_supply(self):
        return sum(self.rw_supply)