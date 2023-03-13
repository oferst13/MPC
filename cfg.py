import numpy as np
import copy
import os
import glob
import pandas as pd
import math
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime

rain_path = 'rain_files/df_rain_files/df_events'
files = glob.glob(rain_path + '/*.csv')
event_df = pd.read_csv(files[72], index_col=False)
rain_header = list(event_df)[1]
rain_array = event_df[rain_header].to_numpy()

dt = 60
rain_dt = 60 * 10
release_dt = 30 * 60
beta = 5 / 4
manning = 0.012
sim_days = min(math.ceil(len(event_df)*rain_dt/(3600*24)) + 0.5, round(len(event_df)*rain_dt/(3600*24)) + 1)
sim_len = int(sim_days * 24 * 60 * 60 / dt)
forecast_hr = 3
forecast_interval = 30 * 60
release_array = (forecast_hr * 60 * 60) / release_dt
forecast_len = int(forecast_hr * 60 * 60 / dt)
forecast_files = 22
t = np.linspace(0, sim_len, num=sim_len + 1)
t = t.astype(int)
hours = t * (dt / 60) / 60
days = hours / 24
collective_hor = True
forecast_hor = 3
if collective_hor:
    prediction_hor = copy.copy(forecast_hor)
    control_hor = copy.copy(forecast_hor)
sample_hr = 1
sample_interval = sample_hr * 60 * 60
sample_len = int(sample_interval / dt)
control_interval = release_dt

forecast_window = int(forecast_hr * 3600 / rain_dt)
window_step = int(forecast_interval / rain_dt)
rain_array = np.concatenate((rain_array, np.zeros(forecast_window - 1)))
rain_array_stacked = sliding_window_view(rain_array, int(forecast_window))[::int(window_step), :]
Cd = 0.5
# Deterministic demands - Change if necessary!
demand_dt = 3 * 60 * 60
demands_3h = np.array([5, 3, 20, 15, 12, 15, 18, 12])
PD = 33
event_start_time = datetime.strptime(event_df['Time'][0].split(' ')[1], '%H:%M:%S').time()
event_start_idx = int(event_start_time.hour*60+event_start_time.minute * (60/dt)) - 1

swmm_files = {(False, 'sim'): 'clustered-no_roof.inp',
              (False, 'real'): 'clustered-no_roof-start.inp',
              (True, 'sim'): 'clustered-no_roof-sim.inp',
              (True, 'real'): 'clustered-no_roof-real.inp'}
