import pandas as pd
from matplotlib import pyplot as plt

results_df = pd.read_csv('results.csv', index_col=False)
ax = results_df.plot(x='Max_EDf5', y='Flow_reduction-perfect', kind='scatter', marker='o', color='xkcd:dark sky blue', label='Perfect', s=30)
results_df.plot(x='Max_EDf5', y='Flow_reduction-plusMin', kind='scatter', marker=(5,1), ax=ax, color='xkcd:goldenrod', label='Depth',s=25)
results_df.plot(x='Max_EDf5', y='Flow_reduction-swap', kind='scatter', marker='+',ax=ax, color='xkcd:kiwi', label='Depth & swap')
ax.legend(title='Forecast type')
plt.xlabel('Max EDf5')
plt.ylabel('Peak Flow Reduction %')
plt.show()

print('end')