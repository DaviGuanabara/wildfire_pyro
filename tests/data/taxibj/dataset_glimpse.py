import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('original-dataset/TAXIBJ2014.grid', sep=",")
df['netflow'] = df["inflow"] - df["outflow"]

print(df.head)
print(df["inflow"].max())
print(df["inflow"].mean())
print(df["outflow"].max())
print(df["outflow"].mean())
print(df["netflow"].max())
print(df["netflow"].min())
print(df["netflow"].mean())

#plt.hist(df["inflow"], bins=30)
#plt.show()
#plt.hist(df["outflow"], bins=30)
#plt.show()
#plt.hist(df["netflow"], bins=30)
#plt.show()

#print("-----ours")
df_ours = pd.read_csv('original-dataset/inflow_14.csv')
print(df_ours.head)

print('number of timesteps:')
print(len(df_ours['t'].unique()))
print('latest timestep')
print(max(df_ours['t']))
#plt.hist(df_ours['t'])
#plt.show()

data_np = df_ours.to_numpy()
print(data_np.shape)

data_np_filtered = data_np[np.where(data_np[:,0] == 241)]
print(data_np_filtered.shape)

data_np_filtered = np.reshape(data_np_filtered, (32, 32, 4))

plt.figure()
plt.imshow(data_np_filtered[:,:,3])
plt.show()

number_of_timesteps = 4368
grid = 32
features = 4

data_reshaped = np.reshape(data_np, (number_of_timesteps, grid, grid, features))
print(data_reshaped.shape)

test_size = 480 # 10 days
val_size = int(480*1.5)
train_size = int(4368 - test_size - val_size)

print('train size, val size, test size: ')
print(train_size, val_size, test_size)
print(train_size/number_of_timesteps, val_size/number_of_timesteps, test_size/number_of_timesteps)

train_np = data_reshaped[:train_size]
val_np = data_reshaped[train_size:(train_size+val_size)]
test_np = data_reshaped[(train_size+val_size):]

print(train_np.shape)
print(val_np.shape)
print(test_np.shape)

#with open(f'taxi_2014_inflow_32_train.npy', 'wb') as f:
#    np.save(f, train_np)
#with open(f'taxi_2014_inflow_32_val.npy', 'wb') as f:
#    np.save(f, val_np)
#with open(f'taxi_2014_inflow_32_test.npy', 'wb') as f:
#    np.save(f, test_np)
