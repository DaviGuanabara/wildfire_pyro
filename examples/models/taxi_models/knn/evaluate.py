import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def plot_map(x, y, true, temp, filename, confidence=False):

    max_limit = np.max(true)
    min_limit = np.min(true)
    
    temp = np.reshape(temp, (grid_side, grid_side))
    
    fig, ax = plt.subplots(1,1)

    if confidence is False:
        cp = ax.contourf(x, y, temp, cmap='coolwarm', vmin=min_limit, vmax=max_limit)
    else:
        cp = ax.contourf(x, y, temp, cmap='coolwarm')
    fig.colorbar(cp) # Add a colorbar to a plot

    ax.set_title('Temperature Map')
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')

    plt.savefig(filename)
    plt.close()

def plot_simple(temp, true, filename):

    max_limit = np.max(true)
    min_limit = np.min(true)

    plt.figure()
    plt.imshow(temp, vmin=min_limit, vmax=max_limit, cmap='RdYlGn_r') # (1, 50, 50, 1)
    plt.colorbar()

    plt.savefig(filename)
    
    plt.close()
    #plt.show()

def predict(features_to_predict, train_features, train_labels):

    temp_list = [] 

    for i in range(n_samples):
        # sample a subset of the data points for bootstrapping
        selection = np.random.choice(np.arange(0,len(train_features)), size=int(len(train_features)*prob), replace=True)

        train_features_subset = train_features[selection] # shape: (len(data_from_db)*prob, 3)
        train_labels_subset = train_labels[selection]

        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        model.fit(train_features_subset, train_labels_subset)

        #if i%10==0:
        #    print("[INFO] {} samples processed".format(i))

        # inference
        temp_list.append(model.predict(features_to_predict))

    temp_array = np.asarray(temp_list) # shape: (n_samples, size_of_mesh*size_of_mesh)

    temp_mean = np.mean(temp_array, axis=0) # shape: (size_of_mesh*size_of_mesh,)
    temp_std = np.std(temp_array, axis=0) # shape: (size_of_mesh*size_of_mesh,)

    return temp_mean, temp_std

def read_data(path):

    data_np = np.load(path)

    print(f'data shape: {data_np.shape}')

    return data_np

def normalize(data, by=2*np.pi):
    data[:,0] /= by
    return data


grid_side = 50

data_type = "fixed"

if data_type == "fixed":
    print("fixed sensors")

    path = "../fixed_test_50.npy"
    # hyperparameters: k neighbors, prob, normalizing factor (larger factor, more importance to time => shorten time dimension)

    # knn variables
    # fixed
    n_samples = 160
    prob = 0.9
    n_neighbors = 3
    past_days_in_prediction = 48

else:
    print("flight sensors")

    path = "../flight_test_50.npy"

    # flight
    n_samples = 160 #90
    prob = 0.7
    n_neighbors = 4
    past_days_in_prediction = 144


data_np = read_data(path)

max_time = np.ceil(np.max(data_np[:,0])) # necessary np.ceil for flight data, because time is a float point and not integer

data1000 = np.load("../../../../data_50_0_1000.npy")
data2000 = np.load('../../../../data_50_1000_2000.npy')
data3000 = np.load('../../../../data_50_2000_3000.npy')
data4000 = np.load('../../../../data_50_3000_4000.npy')
data5000 = np.load('../../../../data_50_4000_5000.npy')
data5664 = np.load('../../../../data_50_5000_5664.npy')

ground_truth = np.concatenate((data1000, data2000, data3000, data4000, data5000, data5664), axis=0)

print(max_time)
data_true = ground_truth[np.where(ground_truth[:,0] == max_time)]
print(data_true.shape)
#print(data_filtered.shape)
y_true = data_true[:,3]

loc = data_true[:,1:3]

mean_squared_error_list = []

data_normalized = normalize(data_np, by=past_days_in_prediction)

data_normalized[:,1] = data_normalized[:,1] / 49.0
data_normalized[:,2] = data_normalized[:,2] / 49.0

train_features = data_normalized[:,:3] # (100, 3)
train_labels = data_normalized[:, -1] # (100,)

t_sample = np.reshape(np.array([max_time/past_days_in_prediction]*2500), (-1,1))


features_to_predict = np.concatenate([t_sample, loc], axis=1)

temp_mean, _ = predict(features_to_predict, train_features, train_labels)
    
mean_squared_error = np.mean(np.square(temp_mean - y_true))

plot_simple(np.reshape(y_true, (50,50)), y_true,  f'test_map_{data_type}/true-c.png')
plot_simple(np.reshape(temp_mean, (50,50)), y_true, f'test_map_{data_type}/predicted-c.png')

#plot_map(x, y, y_true, y_true, f'test_map_{data_type}/true.png')
#plot_map(x, y, y_true, temp_mean, f'test_map_{data_type}/predicted.png')

print(f'RMSE: {np.sqrt(mean_squared_error)}')
print(f'MSE: {(mean_squared_error)}')

