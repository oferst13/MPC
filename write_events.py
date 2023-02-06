import pandas as pd
import numpy as np
from datetime import datetime

diff_event = 6  # difference between event in hours

season_data = pd.read_csv('rain_files/09-10.csv')
season_data['Date'] = season_data['Date'].astype("string")
season_data['Time'] = season_data['Time'].astype("string")
season_data['Time'] = pd.to_datetime(season_data['Date'] + ' ' + season_data['Time'], dayfirst=True)
season_data = season_data.drop(['Date'], axis=1)
rain_dt = (season_data.iloc[1, 0]-season_data.iloc[0, 0]).total_seconds() / 60
rain_header = list(season_data)[1]
rain_array = season_data[rain_header].to_numpy()
print(' ')