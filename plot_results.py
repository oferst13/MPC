import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
plt.rcParams["figure.figsize"] = [9, 6]

results_df = pd.read_csv('results-mae.csv', index_col=False)
results_df['Mean_intens'] = results_df['Total_mm'] / results_df['Duration']
flow_compare = True
if flow_compare:
    q_high = results_df['Flow_from_perfect-swap'].quantile(0.999)
    filtered_df = results_df[(results_df['Flow_from_perfect-swap'] < q_high)]
    filtered_df['Flow_from_perfect-swap'] = filtered_df['Flow_from_perfect-swap']
    filtered_df['Flow_from_perfect-plusMin'] = filtered_df['Flow_from_perfect-plusMin']
#ax = results_df.plot(x='Flow_reduction-perfect', y='Water_reduction-perfect', kind='scatter', marker='o', color='xkcd:dark sky blue', label='Perfect', s=40)
boxplot = False
histo = False
if boxplot:
    plt.rcParams["figure.figsize"] = [4, 6]
    boxprops = dict(color='b', linewidth=3)
    whiskerprops = dict(linewidth=3)
    capsprops = dict(linewidth=3)
    medianprops = dict(linewidth=3, color='g')
    meanpointprops = dict(markersize=7, marker='D')
    fig = results_df.boxplot(column=['Water_from_perfect-plusMin', 'Water_from_perfect-swap'],
                             grid=False, meanline=False, showmeans=True, boxprops=boxprops, whiskerprops=whiskerprops,
                             medianprops=medianprops, capprops=capsprops, meanprops=meanpointprops)
else:
    ax = filtered_df.plot(x='MAE-plusMin', y='Flow_from_perfect-plusMin', kind='scatter', marker=(5,1), color='xkcd:bright orange', label='Depth',s=35)
    #results_df.plot(x='Water_reduction-plusMin', y='Water_reduction-plusMin', kind='scatter', marker=(5,1), ax=ax,
                    #color='xkcd:bright orange', label='Depth', s=35)
    filtered_df.plot(x='MAE-swap', y='Flow_from_perfect-swap', kind='scatter', marker='+',ax=ax, color='xkcd:lightish green', label='Depth & swap',s=30)
    ax.legend(title='Forecast type', fontsize=11)
plt.xlabel('Max EDf5 ' + r'$(\frac{mm}{hr})$', fontsize=12)
#plt.xlabel('Forecast type', fontsize=12)
#plt.ylabel('Water availability reduction %', fontsize=12)
plt.ylabel(r'$\Delta$WR %', fontsize=12)
#plt.xticks([1,2], ['Depth','Depth & swap'], fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.show()

print('end')