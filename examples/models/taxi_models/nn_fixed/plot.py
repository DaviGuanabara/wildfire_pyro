import numpy as np
import matplotlib.pyplot as plt

def plot_simple(temp, true, filename, confidence=False):


    max_limit = np.max(true)
    min_limit = np.min(true)

    plt.figure()
    if confidence:
        plt.imshow(temp) # (1, 50, 50, 1)
    else:
        plt.imshow(temp, vmin=min_limit, vmax=max_limit, cmap='RdYlGn_r') # (1, 50, 50, 1)
    plt.colorbar()

    plt.show()
    print(bas)
    #plt.savefig(filename)
    
    plt.close()

true = np.load("true.npy")
pred = np.load("pred.npy")
conf = np.load("conf.npy")


data1000 = np.load("/Users/leonardo/taxi-bj-data/data_50_0_1000.npy")
data2000 = np.load('/Users/leonardo/taxi-bj-data/data_50_1000_2000.npy')
data3000 = np.load('/Users/leonardo/taxi-bj-data/data_50_2000_3000.npy')
data4000 = np.load('/Users/leonardo/taxi-bj-data/data_50_3000_4000.npy')
data5000 = np.load('/Users/leonardo/taxi-bj-data/data_50_4000_5000.npy')
data5664 = np.load('/Users/leonardo/taxi-bj-data/data_50_5000_5664.npy')

ground_truth = np.concatenate((data1000, data2000, data3000, data4000, data5000, data5664), axis=0)

test_data = np.load("/Users/leonardo/taxi-bj-data/fixed_test_50.npy")

max_time = np.max(test_data[:,0])

data_true = ground_truth[np.where(ground_truth[:,0] == np.ceil(max_time))]
y_ground_truth = data_true[:,3]

plot_simple(np.reshape(true, (50,50)), y_ground_truth, f'test_maps/true_color.png')
plot_simple(np.reshape(pred, (50,50)), y_ground_truth, f'test_maps/predicted_color.png')

