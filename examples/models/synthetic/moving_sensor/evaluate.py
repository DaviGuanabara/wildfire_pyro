import matplotlib.pyplot as plt
from dataset import read_data, PointNeighborhood
import numpy as np
import torch
import os
import json

#from model3 import SpatialRegressor3
from model4 import MultiHeadSpatialRegressor

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


def predict(latitude, longitude, t, n_estimations=1600):

    y_hat_list = []

    x_data = torch.empty(size=(n_estimations, 30, 4))
    mask = torch.empty(size=(n_estimations, 30, 128//4))

    for i in range(n_estimations):
        data_point_dict = test_dataset.generate_test_point(lat=latitude, lon=longitude, t=t)

        x_data[i] = data_point_dict['x_data']
        mask[i] = data_point_dict['mask']

    model.eval()

    with torch.no_grad():

        y_pred = model(
                u=x_data.float(), 
                mask=mask)

    return y_pred.mean().item(), y_pred.std().item()


parameters = {
    "batch_size": None,
    "normalize_timescale": 2*np.pi,
    "learning_rate": 0.5,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "random_noise": False,
    "noise_scale": None,
    "hidden_size": 128, #96,
    "dropout": 0.5,
    "num_epochs": 2500,
    "device": "cpu",
    "last_model": "saved_models/model_6.pt",
    "best_model": "saved_models/best_model_6.pt",
    "plot": "plots/training_6.png",
    "save_every": 100,
    "log_every": 100,
    "n_heads": 4
}


test_data = read_data("../flight_test.csv")


test_dataset = PointNeighborhood(test_data,
                                train=False,
                                hidden=parameters["hidden_size"]//parameters["n_heads"],
                                normalize_time_difference=parameters["normalize_timescale"]) # training the model to predict looking back at this interval


max_time = np.max(test_data[:,0])
#min_time = np.min(test_data[:,0])

#model = SpatialRegressor3(hidden=parameters["hidden_size"], prob=parameters["dropout"])
model = MultiHeadSpatialRegressor(hidden=parameters["hidden_size"], n_head=parameters["n_heads"], prob=parameters["dropout"])

checkpoint = torch.load(parameters["best_model"])

model.load_state_dict(checkpoint['model_state_dict'])

grid_side = 50
lat = np.linspace(0, 1, num=grid_side)
long = np.linspace(0, 1, num=grid_side)

y_true_list = []
y_pred_list = []
squared_error_list = []

i = 0
for latitude in lat:
    for longitude in long:
        
        i += 1
        if i % 100 == 0:
            print(f'prediction number {i}, current error: {np.mean(squared_error_list)}')

        y_true = np.sin(max_time) * latitude + np.cos(max_time) * longitude
        y_hat, _ = predict(latitude=latitude, longitude=longitude, t=max_time, n_estimations=1600)


        squared_error = np.square(y_true - y_hat)

        y_true_list.append(y_true)
        y_pred_list.append(y_hat)

        squared_error_list.append(squared_error)


squared_error_np = np.reshape(np.array(squared_error_list),(-1,1))
mean_squared_error = np.mean(squared_error_np)

print(f"Mean squared error: {mean_squared_error}")

x_mesh, y_mesh = np.meshgrid(lat, long)

x_unraveled = np.reshape(x_mesh, (-1,1))
y_unraveled = np.reshape(y_mesh, (-1,1))

x = np.reshape(x_unraveled, (grid_side, grid_side))
y = np.reshape(y_unraveled, (grid_side, grid_side))

y_true_map = np.reshape(np.array(y_true_list), (grid_side, grid_side))
y_hat_map = np.reshape(np.array(y_pred_list), (grid_side, grid_side))

# z axis boundaries
max = np.max(y_true_list)
min = np.min(y_true_list)

plot_map(x, y, y_true_map, f'test_maps/true.png', max, min)
plot_map(x, y, y_hat_map, f'test_maps/mean.png', max, min)

#data_point_dict = test_dataset.generate_test_point(lat=0.2, lon=0.34, t=max_time)

#print(data_point_dict)
#print(data_point_dict['x_data'].shape) # (30, 4)
#print(data_point_dict['mask'].shape) # (30, 96)



