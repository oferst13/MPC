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
    try:
        first = int(season[0])
    except ValueError:
        print('all metas file exist')
        continue
    temp_df['Season'] = temp_df['Season'].replace('-meta.csv', '', regex=True)
    temp_df['Season'] = temp_df['Season'].replace('-', '-20', regex=True)
    if first < 9:
        temp_df['Season'] = '20' + temp_df['Season'].astype(str)
    else:
        temp_df['Season'] = '19' + temp_df['Season'].astype(str)
    metas_list.append(temp_df)
    print(f'created df with {len(temp_df)} events for {file}')

all_metas = pd.concat(metas_list, axis=0)
write_metas = False
if write_metas:
    all_metas.to_csv(path + '/all_metas.csv', index=False)
plot_3d = True
if plot_3d:
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(projection='3d')
    ax3d.scatter(all_metas['Duration'], all_metas['Total_mm'], all_metas['Max_EDf5'])
    ax3d.set_xlabel('Duration (hr)')
    ax3d.set_ylabel('Total mm')
    ax3d.set_zlabel('Max EDf5(mm/hr)')
    plt.show()

fig, ax = plt.subplots()
ax.scatter(all_metas['Total_mm'], all_metas['Max_EDf5'])

ax.set_xlabel('Total mm')
ax.set_ylabel('Max EDf5 (mm/hr)')

plt.show()

print('')