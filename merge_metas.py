import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

path = 'rain_files/df_rain_files/df_events/meta'
files = glob.glob(path + '/*.csv')

metas_list = []

for file in files:
    season = os.path.basename(file)
    temp_df = pd.read_csv(file, index_col=False)
    temp_df['Season'] = season
    temp_df['Season'] = temp_df['Season'].replace('-meta.csv', '', regex=True)
    temp_df['Season'] = temp_df['Season'].replace('-', '-20', regex=True)
    temp_df['Season'] = '20' + temp_df['Season'].astype(str)
    metas_list.append(temp_df)
    print(f'created df with {len(temp_df)} events for {file}')

all_metas = pd.concat(metas_list, axis=0)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(all_metas['Duration'], all_metas['Total_mm'], all_metas['Max_EDf5'])
ax.set_xlabel('Duration')
ax.set_ylabel('Total_mm')
ax.set_zlabel('Max_EDf5')
plt.show()
print('')