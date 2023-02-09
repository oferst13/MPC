import pandas as pd
import numpy as np
from datetime import datetime
import os


class Event:
    all_events = []

    def __init__(self, first_idx, last_idx):
        self.first_idx = first_idx
        self.last_idx = last_idx
        self.name = None
        self.df = None
        self.meta_df = None
        Event.all_events.append(self)


def close_event(first, last):
    event_array = np.concatenate([[0], rain_array[first:last + 1], [0]])
    tot_mm = round(np.sum(event_array), 2)
    duration = len(event_array) * rain_dt / 60
    return tot_mm, duration

#def find_next_ze


raw_path = 'rain_files'
df_path = raw_path + '/df_rain_files'
diff_event = 12  # min difference between events in hours
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
last_rain = np.max(np.nonzero(rain_array))
i = first_rain
events = []
while i <= last_rain:
    i = i + np.min(np.nonzero(rain_array[i:last_rain + 2]))
    first_i = i
    i = i + np.min(np.nonzero(rain_array[i:last_rain + 2] == 0))
    next_diff = np.sum(rain_array[i:i + int(diff_event * (60 / rain_dt))])
    while next_diff:
        i = i + np.max(np.nonzero(rain_array[i:i + int(diff_event * (60 / rain_dt))]))
        next_diff = np.sum(rain_array[i + 1:i + int(diff_event * (60 / rain_dt))])
    last_i = i
    i += 1
    mm, dur = close_event(first_i, last_i)
    events.append([mm, dur])

print(' ')
