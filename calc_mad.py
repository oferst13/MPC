import pandas as pd
import numpy as np
import cfg
from matplotlib import pyplot as plt


results_df = pd.read_csv('results.csv', index_col=False)
for file in cfg.files:
    event_df = pd.read_csv(file, index_col=False)
    rain_header = list(event_df)[1]
    mad = event_df[rain_header].mad()
    event_idx = results_df.loc[results_df['Dates'] == file.split('\\')[1].split('.')[0]].index[0]
    results_df.at[event_idx, 'MAD'] = mad
print('end')
