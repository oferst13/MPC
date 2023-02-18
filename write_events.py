import pandas as pd
import numpy as np
from datetime import datetime
import os
import math


def close_event(first, last):
    event_array = np.concatenate([[0], rain_array[first:last + 1], [0]])
    tot_mm = round(np.sum(event_array), 2)
    duration = len(event_array) * rain_dt / 60
    return tot_mm, duration


def handle_events(season_events):
    meta_cols = ['Dates', 'Duration', 'Max_EDf5', 'Max30', 'Total_mm']
    season_list = []
    for event in season_events:
        df, str_dates = write_event_to_csv(event[0], event[1])
        meta = calc_meta(df, str_dates)
        season_list.append(meta)
    season_meta = pd.DataFrame(season_list, columns=meta_cols)
    os.makedirs(df_event_path + '/meta', exist_ok=True)
    season_meta.to_csv(df_event_path + '/meta/' + season + '-meta.csv', index=False)


def calc_meta(df, filename):
    duration = (df.Time.iloc[-1] - df.Time.iloc[0]).total_seconds() / 3600
    edf5_window = int(math.ceil((duration * 0.05) * (60 / rain_dt)))
    max30_window = int(30 / rain_dt)
    df['edf5'] = df[rain_header].rolling(edf5_window, min_periods=1).sum()
    df['max30'] = df[rain_header].rolling(max30_window, min_periods=1).sum()
    edf5 = df.edf5.max()
    max30 = df.max30.max()
    tot_mm = df[rain_header].sum()
    # meta_df = pd.DataFrame({'Duration': duration,
    #   'Max30': max30,
    #  'Max_EDf5': edf5,
    #  'Total_mm': tot_mm}, index=[0])
    meta_list = [filename, duration, edf5, max30, tot_mm]
    return meta_list


def write_event_to_csv(first, last):
    event_df = season_data.iloc[int(first - 1):int(last + 2)].copy()
    filename = str(event_df.Time.iloc[0].date()) + ' - ' + str(event_df.Time.iloc[-1].date())
    os.makedirs(df_event_path, exist_ok=True)
    headers = ['Time', rain_header]
    event_df.to_csv(df_event_path + '/' + filename + '.csv', columns=headers, index=False)
    return event_df, filename


raw_path = 'rain_files'
df_path = raw_path + '/df_rain_files'
df_event_path = df_path + '/df_events'
diff_event = 6  # min difference between events in hours
season = '19-20'
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
        i = i + np.max(np.nonzero(rain_array[i: i + int(diff_event * (60 / rain_dt))]))
        next_diff = np.sum(rain_array[i + 1:i + int(diff_event * (60 / rain_dt))])
    last_i = i
    i += 1
    mm, dur = close_event(first_i, last_i)
    events.append([first_i, last_i, mm, dur])
events = np.array(events)
real_events = events[events[:, 2] > 40]
handle_events(real_events)
print(' ')
