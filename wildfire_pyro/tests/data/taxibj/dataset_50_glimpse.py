import numpy as np
import matplotlib.pyplot as plt

path = 'data_50_0_1000.npy'

data = np.load(path)

print(data.shape)
print(max(data[:,1]))

print(data[:,])
data_filtered = data[np.where(data[:,0] == 700)]

data_50 = np.reshape(data_filtered, (-1, 50, 50, 4))

print(data_50.shape)

plt.figure()
plt.imshow(data_50[0,:,:,3], cmap='RdYlGn_r')
plt.colorbar()
plt.show()


data1000 = np.load("/Users/leonardo/taxi-bj-data/data_50_0_1000.npy")
data2000 = np.load('/Users/leonardo/taxi-bj-data/data_50_1000_2000.npy')
data3000 = np.load('/Users/leonardo/taxi-bj-data/data_50_2000_3000.npy')
data4000 = np.load('/Users/leonardo/taxi-bj-data/data_50_3000_4000.npy')
data5000 = np.load('/Users/leonardo/taxi-bj-data/data_50_4000_5000.npy')
data5664 = np.load('/Users/leonardo/taxi-bj-data/data_50_5000_5664.npy')

ground_truth = np.concatenate((data1000, data2000, data3000, data4000, data5000, data5664), axis=0)

plt.figure()
plt.hist(ground_truth[:,3], bins=20, color='gray', edgecolor='k', alpha=0.65)
plt.axvline(ground_truth[:,3].mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()
