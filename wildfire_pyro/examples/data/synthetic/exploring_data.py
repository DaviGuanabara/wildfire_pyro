import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# path = "fixed_test.csv"
path = "flight_test.csv"

df = pd.read_csv(path)

print(df)
print(df['t'])

print('time')
print(f"min time: {df['t'].min()}")
print(f"max time: {df['t'].max()}")
print(f"max time normalized by pi: {df['t'].max() / (np.pi)}")

plt.figure()
plt.hist(df['t'])
plt.show()

plt.figure()
plt.plot(df['lat'], df['lon'], '.')
plt.show()

plt.figure()
plt.hist(df['y'])
plt.show()

print(len(df))
