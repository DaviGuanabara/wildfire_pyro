import matplotlib.pyplot as plt
from dataset import read_data, PointNeighborhood
import numpy as np
import torch
import os
import json

#from model4 import MultiHeadSpatialRegressor
from model3c import SpatialRegressor3
from sklearn.preprocessing import MinMaxScaler

def plot_map(x, y, temp, filename, max=None, min=None, confidence=False):

    fig, ax = plt.subplots(1,1)

    if confidence is False:
        cp = ax.contourf(x, y, temp, cmap='coolwarm', vmin=min, vmax=max)
    else:
        cp = ax.contourf(x, y, temp, cmap='coolwarm')
    fig.colorbar(cp) # Add a colorbar to a plot

    ax.set_title('Temperature Map')
    ax.set_ylabel('latitude')
    ax.set_xlabel('longitude')

    plt.savefig(filename)
    plt.close()

def plot_simple(temp, true, filename, confidence=False):


    max_limit = np.max(true)
    min_limit = np.min(true)

    plt.figure()
    if confidence:
        plt.imshow(temp) # (1, 50, 50, 1)
    else:
        plt.imshow(temp, vmin=min_limit, vmax=max_limit) # (1, 50, 50, 1)
    plt.colorbar()

    plt.savefig(filename)
    
    plt.close()

def predict(loc, t, n_estimations=1600):

    x_data = torch.empty(size=(n_estimations, 50, 4))
    mask = torch.empty(size=(n_estimations, 50, 128))#256//8))

    for i in range(n_estimations):
        data_point_dict = test_dataset.generate_test_point(lat=loc[0]/49.0, lon=loc[1]/49.0, t=t)

        x_data[i] = data_point_dict['x_data']
        mask[i] = data_point_dict['mask']

    model.eval()

    with torch.no_grad():

        y_pred = model(
                u=x_data.float(), 
                mask=mask)

    return y_pred.mean().item(), y_pred.std().item()


parameters = {
    "batch_size": 2048,
    "normalize_timescale": 480,
    "learning_rate": 0.0001,#0.1,
    "weight_decay": 1e-5,
    "momentum": 0.9,
    "random_noise": False,
    "noise_scale": None,
    "hidden_size": 128, #256, #96,
    "dropout": 0.1,
    "num_epochs": 2500,
    "device": "cpu",
    "last_model": "saved_models/model_1.pt",
    "best_model": "saved_models/best_model_1.pt", #"/Users/leonardo/taxi-bj-data/taxi-flight/models/best_model_4_tanh.pt",
    "plot": "plots/training_1.png",
    "save_every": 1,
    "log_every": 1,
    "n_heads": 1 
}


data1000 = np.load("/Users/leonardo/taxi-bj-data/data_50_0_1000.npy")
data2000 = np.load('/Users/leonardo/taxi-bj-data/data_50_1000_2000.npy')
data3000 = np.load('/Users/leonardo/taxi-bj-data/data_50_2000_3000.npy')
data4000 = np.load('/Users/leonardo/taxi-bj-data/data_50_3000_4000.npy')
data5000 = np.load('/Users/leonardo/taxi-bj-data/data_50_4000_5000.npy')
data5664 = np.load('/Users/leonardo/taxi-bj-data/data_50_5000_5664.npy')

ground_truth = np.concatenate((data1000, data2000, data3000, data4000, data5000, data5664), axis=0)

test_data = read_data("/Users/leonardo/taxi-bj-data/flight_test_50.npy")
train_data = read_data("/Users/leonardo/taxi-bj-data/flight_train_50.npy")

output_scaler = MinMaxScaler().fit(train_data[:,3].reshape(-1,1))
#output_scaler = None

#test_data[:,1] = test_data[:,1] / 49.0
#test_data[:,2] = test_data[:,2] / 49.0

test_dataset = PointNeighborhood(test_data,
                                train=False,
                                hidden=parameters["hidden_size"]//parameters["n_heads"],
                                normalize_time_difference=parameters["normalize_timescale"],
                                output_scaler=output_scaler,
                                min_neighbors=20,
                                max_neighbors=50) # training the model to predict looking back at this interval

max_time = np.max(test_data[:,0])
#min_time = np.min(test_data[:,0])

data_true = ground_truth[np.where(ground_truth[:,0] == np.ceil(max_time))]
y_ground_truth = data_true[:,3]
loc = data_true[:,1:3] * 49.

print(max_time)
print(ground_truth.shape)
print(data_true.shape)

model = SpatialRegressor3(hidden=parameters["hidden_size"], prob=parameters["dropout"])
#model = MultiHeadSpatialRegressor(hidden=parameters["hidden_size"], n_head=parameters["n_heads"], prob=parameters["dropout"])

#checkpoint = torch.load(parameters["best_model"], map_location=torch.device('cpu'))
checkpoint = torch.load(parameters["best_model"])

model.load_state_dict(checkpoint['model_state_dict'])

grid_side = 50

y_true_list = []
y_pred_list = []
y_conf_list = []
squared_error_list = []

i = 0
# make this loop be using the loc coordinates
for location, y_true in zip(loc, y_ground_truth):

    i += 1
    if i % 100 == 0:
        print(f'prediction number {i}, current error: {np.mean(squared_error_list)}')

    #y_true = np.sin(max_time) * latitude + np.cos(max_time) * longitude
    y_hat_scaled, y_conf = predict(loc=location, t=max_time, n_estimations=1600)

    #if output_scaler is not None:
    #    y_hat = output_scaler.inverse_transform(np.array([[y_hat_scaled]]))
    #else:
    #    y_hat = y_hat_scaled
    
    y_hat = y_hat_scaled

    squared_error = np.square(y_true - float(y_hat))

    y_true_list.append(y_true)
    y_pred_list.append(y_hat)
    y_conf_list.append(y_conf)

    squared_error_list.append(squared_error)


squared_error_np = np.reshape(np.array(squared_error_list),(-1,1))
mean_squared_error = np.mean(squared_error_np)

print(f"Mean squared error: {mean_squared_error}")
print(f'RMSE: {np.sqrt(mean_squared_error)}')

plot_simple(np.reshape(np.array(y_true_list), (50,50)), y_ground_truth, f'test_maps/true.png')
plot_simple(np.reshape(np.array(y_pred_list), (50,50)), y_ground_truth, f'test_maps/predicted.png')
plot_simple(np.reshape(np.array(y_conf_list), (50,50)), y_ground_truth, f'test_maps/confidence.png', confidence=True)
plot_simple(np.reshape(np.array(squared_error_list), (50,50)), y_ground_truth, f'test_maps/error.png', confidence=True)

def save_array(path, arr):
    with open(path, 'wb') as f:
        np.save(f, arr)

save_array("true.npy", np.array(y_true_list))
save_array("pred.npy", np.array(y_pred_list))
save_array("conf.npy", np.array(y_conf_list))


