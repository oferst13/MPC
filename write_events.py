import pandas as pd
import numpy as np
from datetime import datetime
import os


class Event:
    def __init__(self, first_idx, last_idx):
        self.first_idx = first_idx
        self.last_idx = last_idx
        self.name = None
        self.df = None
        self.meta_df = None


raw_path = 'rain_files'
df_path = raw_path + '/df_rain_files'
diff_event = 6  # difference between event in hours
season = '09-10'
rain_file = season + '.csv'
rain_df_file = df_path + '/' + season + '-df.csv'

if os.path.isfile(rain_df_file):
    print('rain df file exists')
    season_data = pd.read_csv(rain_df_file)
    season_data['Time'] = pd.to_datetime(season_data['Time'], dayfirst=True)
else:
    print('rain df file missing. writing to df...')
    season_data = pd.read_csv('rain_files' + '/' + rain_file)
    season_data['Date'] = season_data['Date'].astype("string")
    season_data['Time'] = season_data['Time'].astype("string")
    season_data['Time'] = pd.to_datetime(season_data['Date'] + ' ' + season_data['Time'], dayfirst=True)
    season_data = season_data.drop(['Date'], axis=1)
    os.makedirs(df_path, exist_ok=True)
    write_df_to_csv = True
    if write_df_to_csv:
        try:
            season_data.to_csv(df_path + '/' + season + '-df.csv', index=False, mode='x')
            print('Season rainfile saved as df')
        except FileExistsError:
            print('File exists, no overwriting made')
rain_dt = (season_data.iloc[1, 0] - season_data.iloc[0, 0]).total_seconds() / 60
rain_header = list(season_data)[1]
rain_array = season_data[rain_header].to_numpy()
first_rain = np.min(np.nonzero(rain_array))
print(' ')
