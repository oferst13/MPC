import pandas as pd
import cfg
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error


def set_forecast_idx(first, last, diff):
    return np.arange(first, last, diff)


results_df = pd.read_csv('results.csv', index_col=False)
rain_path = 'rain_files/df_rain_files/df_events'
forecast_path = 'rain_files/Forecasts/'
for file in cfg.files:
    for suff in ['plusMin', 'swap']:
        forecast_mode = '-' + suff + '.csv'
        event_df = pd.read_csv(file, index_col=False)
        rain_header = list(event_df)[1]
        rain_array = event_df[rain_header].to_numpy()
        event_dates = file.split('\\')[1].split('.')[0]
        forecast_array = np.genfromtxt(forecast_path + event_dates + forecast_mode, delimiter=',')
        num_forecasts = len(forecast_array)
        forecast_indices = set_forecast_idx(0, num_forecasts, int(cfg.sample_interval / cfg.forecast_interval))
        to_compare = np.zeros((len(forecast_indices), len(forecast_array[0, :]), 2))
        for target_idx, idx in enumerate(forecast_indices):
            to_compare[target_idx, :, 0] = forecast_array[idx, :]
            start_idx_rain = int(target_idx * (cfg.sample_interval / cfg.rain_dt))
            rain_to_write = rain_array[start_idx_rain:start_idx_rain+18]
            to_compare[target_idx, :len(rain_to_write), 1] = rain_to_write
        forecast_flat = to_compare[:, :, 0]
        rain_flat = to_compare[:, :, 1]
        mae = mean_absolute_error(rain_flat, forecast_flat)
        event_idx = results_df.loc[results_df['Dates'] == file.split('\\')[1].split('.')[0]].index[0]
        results_df.at[event_idx, 'MAE-' + suff] = mae
rain_hours = np.linspace(0, int(cfg.sim_days * 24), int(cfg.sim_days * 24 * 3600 / cfg.rain_dt) + 1,
                         dtype='longfloat')
print(' ')
