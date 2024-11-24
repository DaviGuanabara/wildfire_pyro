import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./flight.csv')
data_np = df.to_numpy()

plt.figure()
plt.hist(data_np[:,3], bins=20, color='gray', edgecolor='k', alpha=0.65)
plt.axvline(data_np[:,3].mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()
