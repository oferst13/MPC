import copy

import numpy as np
import cfg
import os
import glob
import pandas as pd
import math
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime

swap_probs = [0.2, 0.3, 0.4]
vol_probs = [0.2, 0.3, 0.4]
swap = True
if swap:
    suffix = '-swap.csv'
else:
    suffix = '-plusMin.csv'
rain_path = 'rain_files/df_rain_files/df_events'
forecast_path = 'rain_files/Forecasts/'
files = glob.glob(rain_path + '/*.csv')
for file in files:
    event_df = pd.read_csv(file, index_col=False)
    rain_header = list(event_df)[1]
    rain_array = event_df[rain_header].to_numpy()
    event_dates = file.split('\\')[1].split('.')[0]
    forecast_window = int(cfg.forecast_hr * 3600 / cfg.rain_dt)
    window_step = int(cfg.forecast_interval / cfg.rain_dt)
    rain_array = np.concatenate((rain_array, np.zeros(forecast_window - 1)))
    rain_array_stacked = sliding_window_view(rain_array, int(forecast_window))[::int(window_step), :]
    forecast_rain = np.zeros_like(rain_array_stacked)
    for idx, row in enumerate(rain_array_stacked):
        row_to_write = copy.copy(row)
        for i in range(len(row)):
            row_to_write[i] += row_to_write[i] * np.random.uniform(-vol_probs[i // 6], vol_probs[i // 6])
            row_to_write[i] = np.round_(row_to_write[i], 2)
            if swap:
                if np.random.rand() < swap_probs[i // 6] and i > 0:
                    try:
                        row_to_write[i], row_to_write[i + 1] = row_to_write[i + 1], row_to_write[i]
                        print(i, ' swap')
                    except IndexError:
                        print('last')
        forecast_rain[idx] = row_to_write
    forecast_df = pd.DataFrame(forecast_rain)
    forecast_df.to_csv(forecast_path + event_dates + suffix, index=False, header=False)