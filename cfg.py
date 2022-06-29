import numpy as np
import copy

dt = 60
rain_dt = 60 * 10
beta = 5 / 4
manning = 0.012
sim_days = 1
sim_len = int(sim_days * 24 * 60 * 60 / dt)
t = np.linspace(0, sim_len, num=sim_len + 1)
t = t.astype(int)
hours = t * (dt / 60) / 60
days = hours / 24
collective_hor = True
forecast_hor = 3
if collective_hor:
    prediction_hor = copy.copy(forecast_hor)
    control_hor = copy.copy(forecast_hor)

Cd = 0.6
# Deterministic demands - Change if necessary!
demand_dt = 3 * 60 * 60
demands_3h = np.array([5, 3, 20, 15, 12, 15, 18, 12])
PD = 33


