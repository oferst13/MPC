import pandas as pd
import cfg
from matplotlib import pyplot as plt
import numpy as np

rain_path = 'rain_files/df_rain_files/df_events'
forecast_path = 'rain_files/Forecasts/'
file = cfg.files[0]
event_df = pd.read_csv(file, index_col=False)
rain_header = list(event_df)[1]
rain_array = event_df[rain_header].to_numpy()
event_dates = file.split('\\')[1].split('.')[0]
forecast_df = pd.read_csv(forecast_path + event_dates + '-swap.csv')
forecast_array = forecast_df[rain_header].to_numpy()
rain_hours = np.linspace(0, int(cfg.sim_days * 24), int(cfg.sim_days * 24 * 3600 / cfg.rain_dt) + 1,
                             dtype='longfloat')
print(' ')