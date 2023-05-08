import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [9, 6]

results_df = pd.read_csv('results.csv', index_col=False)
ax = results_df.plot(x='Max_EDf5', y='Water_reduction-perfect', kind='scatter', marker='o', color='xkcd:dark sky blue', label='Perfect', s=40)
results_df.plot(x='Max_EDf5', y='Water_reduction-plusMin', kind='scatter', marker=(5,1), ax=ax, color='xkcd:bright orange', label='Depth',s=35)
results_df.plot(x='Max_EDf5', y='Water_reduction-swap', kind='scatter', marker='+',ax=ax, color='xkcd:lightish green', label='Depth & swap',s=30)
ax.legend(title='Forecast type', fontsize=11)
plt.xlabel('Max EDf5 ' + r'$(\frac{mm}{hr})$', fontsize=12)
plt.ylabel('Rainwater Availability Reduction %', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.show()

print('end')