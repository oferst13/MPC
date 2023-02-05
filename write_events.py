import pandas as pd
import numpy as np
from datetime import datetime


season_data = pd.read_csv('rain_files/09-10.csv')
season_data['Date'] = season_data['Date'].astype("string")
season_data['Time'] = season_data['Time'].astype("string")
season_data['Time'] = pd.to_datetime(season_data['Date'] + ' ' + season_data['Time'], dayfirst=True)
season_data = season_data.drop(['Date'], axis=1)
print(' ')