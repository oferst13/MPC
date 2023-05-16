import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
plt.rcParams["figure.figsize"] = [9, 6]

results_df = pd.read_csv('results-mae.csv', index_col=False)
#ax = results_df.plot(x='Max_EDf5', y='Water_reduction-perfect', kind='scatter', marker='o', color='xkcd:dark sky blue', label='Depth', s=40)
boxplot = False
if boxplot:
    boxprops = dict(color='b', linewidth=3)
    whiskerprops = dict(linewidth=3)
    capsprops = dict(linewidth=3)
    medianprops = dict(linewidth=3, color='g')
    meanpointprops = dict(markersize=10, marker='D')
    fig = results_df.boxplot(column=['Water_reduction-perfect', 'Water_reduction-plusMin', 'Water_reduction-swap'],
                             grid=False, meanline=False, showmeans=True, boxprops=boxprops, whiskerprops=whiskerprops,
                             medianprops=medianprops, capprops=capsprops, meanprops=meanpointprops)
else:
    ax = results_df.plot(x='Max_EDf5', y='Water_delta-plusMin', kind='scatter', marker=(5,1), color='xkcd:bright orange', label='Depth',s=35)
    #results_df.plot(x='Max_EDf5', y='Water_reduction-plusMin', kind='scatter', marker=(5,1), ax=ax,
                    #color='xkcd:bright orange', label='Depth & swap', s=35)
    results_df.plot(x='Max_EDf5', y='Water_delta-swap', kind='scatter', marker='+',ax=ax, color='xkcd:lightish green', label='Depth & swap',s=30)
    ax.legend(title='Forecast type', fontsize=11)
plt.xlabel('Max EDf5 ' + r'$(\frac{mm}{hr})$', fontsize=12)
#plt.xlabel('Forecast type', fontsize=12)
#plt.ylabel(r'$\Delta$WR %', fontsize=12)
plt.ylabel('Rainwater availability reduction %', fontsize=12)
#plt.xticks([1,2,3], ['Perfect','Depth','Depth & swap'], fontsize=11)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.show()

print('end')