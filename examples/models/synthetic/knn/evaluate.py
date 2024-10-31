
import matplotlib
#matplotlib.use("Agg")

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
        cp = ax.contourf(x, y, temp.T, cmap='coolwarm', vmin=min_limit, vmax=max_limit)
    else:
        cp = ax.contourf(x, y, temp.T, cmap='coolwarm')
    fig.colorbar(cp) # Add a colorbar to a plot

    ax.set_title('Temperature Map')
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')

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

def plot_ground_truth(t_interval, i):

    for e, time in enumerate(t_interval):
        y_true = np.sin(time) * location[:,0] + np.cos(time) * location[:,1]

        y_true_map = np.reshape(y_true, (size_of_mesh,size_of_mesh))

        plot_map(x, y, y_true_map, f'train_data_plots/{i+e}_true.png', max_limit, min_limit)
    return

def read_data(path):

    df = pd.read_csv(path)
    data_np = df.to_numpy()

    print(f'data shape: {data_np.shape}')

    return data_np

def normalize(data, by=2*np.pi):
    data[:,0] /= by
    return data


grid_side = 50

lat = np.linspace(0, 1, num=grid_side)
long = np.linspace(0, 1, num=grid_side)

x_mesh, y_mesh = np.meshgrid(lat, long)
x_unraveled = np.reshape(x_mesh, (-1,1))
y_unraveled = np.reshape(y_mesh, (-1,1))
location = np.array((x_unraveled, y_unraveled))
location = np.squeeze(location)
location = location.T
print(location.shape)

x = np.reshape(x_unraveled, (grid_side, grid_side))
y = np.reshape(y_unraveled, (grid_side, grid_side))

#data_type = "fixed"
data_type = "flight"

if data_type == "fixed":
    print("fixed sensors")

    path = "../fixed_test.csv"
    # hyperparameters: k neighbors, prob, normalizing factor (larger factor, more importance to time => shorten time dimension)

    # knn variables
    # fixed
    n_samples = 160
    prob = 0.6
    n_neighbors = 3
    normalize_by = 1.0

else:
    print("flight sensors")

    path = "../flight_test.csv"

    # flight
    n_samples = 300#160 #90
    prob = 0.3
    n_neighbors = 8
    normalize_by = 1.0 #2.174
    # in this config 0.440 mse

data_np = read_data(path)

#

print(np.sort(data_np[:,0]))

plt.figure()
#plt.hist(data_np[:,-1])
#plt.hist(data_np[:,0])
#plt.plot(data_np[:,1], data_np[:,2], '.')
pz = np.array((data_np[:,0]-np.min(data_np[:,0]))/(np.max(data_np[:,0])-np.min(data_np[:,0])))
colors = np.array([[176/255, 10/255, 20/255, 1.        ]]*(len(data_np)))
colors[:,-1] = pz
plt.scatter(data_np[:,1], data_np[:,2], color=colors, marker='o')
plt.show()
print(bas)
#

mean_squared_error_list = []

max_time = np.max(data_np[:,0])
max_time = 24.10

data_normalized = normalize(data_np, by=normalize_by)

y_true = np.sin(max_time) * location[:,0] + np.cos(max_time) * location[:,1]

train_features = data_normalized[:,:3] # (100, 3)
train_labels = data_normalized[:, -1] # (100,)

t_sample = [max_time/normalize_by]*2500

features_to_predict = np.column_stack([t_sample, location])

temp_mean, temp_confidence = predict(features_to_predict, train_features, train_labels)
    
mean_squared_error = (np.mean(np.square(temp_mean - y_true)))

plot_map(x, y, y_true, y_true, f'test_map_{data_type}/true.png')
plot_map(x, y, y_true, temp_mean, f'test_map_{data_type}/predicted.png')
plot_map(x, y, y_true, temp_confidence, f'test_map_{data_type}/confidence.png', confidence=True)

abs_error = np.abs(temp_mean - y_true)
plot_map(x, y, y_true, abs_error, f'test_map_{data_type}/error.png', confidence=True)

print(f'RMSE: {np.sqrt(mean_squared_error)}')
print(f'MSE: {(mean_squared_error)}')

