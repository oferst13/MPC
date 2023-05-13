import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
plt.rcParams["figure.figsize"] = [9, 6]

results_df = pd.read_csv('results.csv', index_col=False)
#ax = results_df.plot(x='Max_EDf5', y='Flow_delta-plusMin', kind='scatter', marker='o', color='xkcd:dark sky blue', label='Depth', s=40)
ax = results_df.plot(x='Max_EDf5', y='try', kind='scatter', marker=(5,1), color='xkcd:bright orange', label='Depth',s=35)
results_df.plot(x='Max_EDf5', y='try1', kind='scatter', marker='+',ax=ax, color='xkcd:lightish green', label='Depth & swap',s=30)
ax.legend(title='Forecast type', fontsize=11)
plt.xlabel('Max EDf5 ' + r'$(\frac{mm}{hr})$', fontsize=12)
plt.ylabel(r'$\Delta$FR %', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.show()

print('end')