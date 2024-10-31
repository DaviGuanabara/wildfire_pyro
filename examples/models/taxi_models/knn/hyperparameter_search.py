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

def plot_simple(temp, filename):

    plt.figure()
    plt.imshow(temp) # (1, 50, 50, 1)
    plt.savefig(filename)
    plt.close()


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

#path = "../fixed_train_50.npy"
path = "../flight_train_50.npy"
data_np = read_data(path)

data1000 = np.load("../../../../data_50_0_1000.npy")
data2000 = np.load('../../../../data_50_1000_2000.npy')
data3000 = np.load('../../../../data_50_2000_3000.npy')
data4000 = np.load('../../../../data_50_3000_4000.npy')
data5000 = np.load('../../../../data_50_4000_5000.npy')
data5664 = np.load('../../../../data_50_5000_5664.npy')

ground_truth = np.concatenate((data1000, data2000, data3000, data4000, data5000, data5664), axis=0)

print(f'train shape: {data_np.shape}')
print(f'ground truth shape: {ground_truth.shape}')

# hyperparameters: k neighbors, prob, normalizing factor (larger factor, more importance to time => shorten time dimension)
# knn variables
n_samples = 160
prob = 0.7
n_neighbors = 4
past_days_in_prediction = 144

#for normalize_by in np.linspace(1, 2*np.pi, 10):
#    print('normalizing factor: ')
#    print(normalize_by)

#for n_neighbors in range(3, 20, 1):
#    print(f'n neighbors {n_neighbors}')

for prob in np.linspace(0.1, 0.9, num=9):
    print(f'prob: {prob}')

#for past_days_in_prediction in range(48, (48*15), 48):
#    print(f'days history in prediction: {past_days_in_prediction}')

#for n_samples in range(50, 300, 20):
#    print(f'n_samples: {n_samples}')

    mean_squared_error_list = []

    # 480 represent 10 days
    times_to_predict = np.linspace(past_days_in_prediction, 4463, 25)

    for i, t in enumerate(times_to_predict): # predict ten complete maps
       
        timestep = int(t)

        # filter only data points in the past that are less than 10 days away from the present time
        data_filtered = data_np[np.where((data_np[:,0] <= t) & (np.abs(data_np[:,0]-t) <= past_days_in_prediction))]

        data_normalized = normalize(data_filtered, by=past_days_in_prediction)

        # normalize coordinates between 0 and 1
        data_normalized[:,1] = data_normalized[:,1] / 49.0
        data_normalized[:,2] = data_normalized[:,2] / 49.0
        
        #least_recent_timestep = min(data_normalized[:,0])
        #most_recent_timestep = max(data_normalized[:,0])

        data_true = ground_truth[np.where(ground_truth[:,0] == timestep)]

        #print(data_filtered.shape)
        y_true = data_true[:,3]

        loc = data_true[:,1:3]

        #print(y_true.shape)

        if len(y_true) == 0:
            continue

        #y_true = np.reshape(y_true, (-1, 50, 50, 4))
        #y_true = np.squeeze(y_true, axis=0)

        train_features = data_normalized[:,:3] # (100, 3)
        train_labels = data_normalized[:, -1] # (100,)

        #t_sample = np.reshape(np.array([most_recent_timestep]*2500), (-1,1))
        t_sample = np.reshape(np.array([t/past_days_in_prediction]*2500), (-1,1))

        #print(t_sample)
        #print(train_features.shape)
        #print(train_features.shape)
        #print(bas)

        features_to_predict = np.concatenate([t_sample, loc], axis=1)

        temp_mean, _ = predict(features_to_predict, train_features, train_labels)

        root_mean_squared_error = np.sqrt(np.mean(np.square(temp_mean - y_true)))

        mean_squared_error_list.append(root_mean_squared_error)

        #plot_simple(np.reshape(y_true, (50,50)), f'train_eval_plots_flight/{i}_true.png')
        #plot_simple(np.reshape(temp_mean, (50,50)), f'train_eval_plots_flight/{i}_predicted.png')

    mean_error = np.mean(mean_squared_error_list)
    print(f'RMSE: {mean_error}')
    #break
    #print(f'MSE: {mean_error}')



