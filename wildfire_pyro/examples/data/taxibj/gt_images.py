from sklearn.neighbors import KNeighborsRegressor
from model3c import SpatialRegressor3
import torch
from dataset import PointNeighborhood
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")


def predict_knn(features_to_predict, train_features,
                train_labels, n_samples=600, prob=0.7, n_neighbors=4):

    temp_list = []

    for i in range(n_samples):
        # sample a subset of the data points for bootstrapping
        selection = np.random.choice(np.arange(0, len(train_features)), size=int(
            len(train_features) * prob), replace=True)

        # shape: (len(data_from_db)*prob, 3)
        train_features_subset = train_features[selection]
        train_labels_subset = train_labels[selection]

        model = KNeighborsRegressor(
            n_neighbors=n_neighbors, weights='distance')
        model.fit(train_features_subset, train_labels_subset)

        # if i%10==0:
        #    print("[INFO] {} samples processed".format(i))

        # inference
        temp_list.append(model.predict(features_to_predict))

    # shape: (n_samples, size_of_mesh*size_of_mesh)
    temp_array = np.asarray(temp_list)

    # shape: (size_of_mesh*size_of_mesh,)
    temp_mean = np.mean(temp_array, axis=0)
    # shape: (size_of_mesh*size_of_mesh,)
    temp_std = np.std(temp_array, axis=0)

    return temp_mean, temp_std


def normalize(data, by=144):
    data[:, 0] /= by
    return data


def predict(loc, t, n_estimations=1600):

    x_data = torch.empty(size=(n_estimations, 50, 4))
    mask = torch.empty(size=(n_estimations, 50, 128))  # 256//8))

    for i in range(n_estimations):
        data_point_dict = test_dataset.generate_test_point(
            lat=loc[0] / 49.0, lon=loc[1] / 49.0, t=t)

        x_data[i] = data_point_dict['x_data']
        mask[i] = data_point_dict['mask']

    model.eval()

    with torch.no_grad():

        y_pred = model(
            u=x_data.float(),
            mask=mask)

    return y_pred.mean().item(), y_pred.std().item()


gt = False
sensor = False
nn = True
knn = False

data1000 = np.load("/Users/leonardo/taxi-bj-data/data_50_0_1000.npy")
data2000 = np.load('/Users/leonardo/taxi-bj-data/data_50_1000_2000.npy')
data3000 = np.load('/Users/leonardo/taxi-bj-data/data_50_2000_3000.npy')
data4000 = np.load('/Users/leonardo/taxi-bj-data/data_50_3000_4000.npy')
data5000 = np.load('/Users/leonardo/taxi-bj-data/data_50_4000_5000.npy')
data5664 = np.load('/Users/leonardo/taxi-bj-data/data_50_5000_5664.npy')

ground_truth = np.concatenate(
    (data1000, data2000, data3000, data4000, data5000, data5664), axis=0)

max_limit = np.max(ground_truth[:, 3])
min_limit = np.min(ground_truth[:, 3])

time = ground_truth[:, 0]
time_set = set(time)
time_list = list(time_set)
time_list.sort()
last_25_days = time_list[-1200:]

val_sensor = np.load("/Users/leonardo/taxi-bj-data/flight_val_50.npy")
test_sensor = np.load('/Users/leonardo/taxi-bj-data/flight_test_50.npy')

print(last_25_days[:10])
print(last_25_days[-10:])
sensor_data = np.concatenate((val_sensor, test_sensor), axis=0)
print(sensor_data.shape)


data_true = ground_truth[np.where(ground_truth[:, 0] == np.ceil(max_limit))]

y_true = data_true[:, 3]
loc = data_true[:, 1:3]


if nn:
    parameters = {
        "batch_size": 2048,
        "normalize_timescale": 480,
        "learning_rate": 0.0001,  # 0.1,
        "weight_decay": 1e-5,
        "momentum": 0.9,
        "random_noise": False,
        "noise_scale": None,
        "hidden_size": 128,  # 256, #96,
        "dropout": 0.1,
        "num_epochs": 2500,
        "device": "cpu",
        "last_model": "saved_models/model_1.pt",
        # "/Users/leonardo/taxi-bj-data/taxi-flight/models/best_model_4_tanh.pt",
        "best_model": "/Users/leonardo/wildfire-sinusoidal/taxi_nn_flight/saved_models/best_model_1.pt",
        "plot": "plots/training_1.png",
        "save_every": 1,
        "log_every": 1,
        "n_heads": 1
    }
    train_data = np.load("/Users/leonardo/taxi-bj-data/flight_train_50.npy")
    output_scaler = MinMaxScaler().fit(train_data[:, 3].reshape(-1, 1))

    model = SpatialRegressor3(
        hidden=parameters["hidden_size"], prob=parameters["dropout"])

    checkpoint = torch.load(parameters["best_model"])

    model.load_state_dict(checkpoint['model_state_dict'])

    loc = data_true[:, 1:3] * 49.

for index, i in enumerate(last_25_days):

    if nn:
        print(f'index: {index}')

        test_data = sensor_data[np.where(sensor_data[:, 0] <= i)]
        y_pred_list = []
        y_conf_list = []

        if len(test_data) > 60:

            test_dataset = PointNeighborhood(test_data,
                                             train=False,
                                             hidden=parameters["hidden_size"] // parameters["n_heads"],
                                             normalize_time_difference=parameters["normalize_timescale"],
                                             output_scaler=output_scaler,
                                             min_neighbors=20,
                                             max_neighbors=50)  # training the model to predict looking back at this interval

            max_time = np.max(test_data[:, 0])
            i = 0
            # make this loop be using the loc coordinates
            for location in loc:
                y_hat, y_conf = predict(
                    loc=location, t=max_time, n_estimations=1600)

                print(y_hat)
                y_pred_list.append(y_hat)
                y_conf_list.append(y_conf)

            plt.figure()
            plt.imshow(np.reshape(np.array(y_pred_list), (50, 50)),
                       # (1, 50, 50, 1)
                       vmin=min_limit, vmax=max_limit, cmap='RdYlGn_r')
            plt.colorbar()
            plt.savefig(f'video-images/predictions/prediction-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()

            plt.figure()
            plt.imshow(np.reshape(np.array(y_conf_list), (50, 50)),
                       # (1, 50, 50, 1)
                       vmin=min_limit, vmax=max_limit, cmap='Reds')
            plt.colorbar()
            plt.savefig(f'video-images/confidence/confidence-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()
        else:
            plt.figure()
            plt.imshow(np.zeros((50, 50)), vmin=min_limit,
                       vmax=max_limit, cmap='RdYlGn_r')  # (1, 50, 50, 1)
            plt.colorbar()
            plt.savefig(f'video-images/predictions/prediction-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()

            plt.figure()
            plt.imshow(np.zeros((50, 50)), vmin=min_limit,
                       vmax=max_limit, cmap='Reds')  # (1, 50, 50, 1)
            plt.colorbar()
            plt.savefig(f'video-images/confidence/confidence-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()

    if knn:

        test_data = sensor_data[np.where(sensor_data[:, 0] <= i)]
        y_pred_list = []
        y_conf_list = []

        if len(test_data) > 50:

            max_time = np.max(test_data[:, 0])
            data_normalized = normalize(test_data)

            data_normalized[:, 1] = data_normalized[:, 1] / 49.0
            data_normalized[:, 2] = data_normalized[:, 2] / 49.0

            train_features = data_normalized[:, :3]  # (100, 3)
            train_labels = data_normalized[:, -1]  # (100,)

            t_sample = np.reshape(np.array([max_time / 144] * 2500), (-1, 1))

            features_to_predict = np.concatenate([t_sample, loc], axis=1)

            temp_mean, temp_confidence = predict_knn(
                features_to_predict, train_features, train_labels)

            plt.figure()
            plt.imshow(np.reshape(temp_mean, (50, 50)), vmin=min_limit,
                       vmax=max_limit, cmap='RdYlGn_r')  # (1, 50, 50, 1)
            plt.colorbar()
            plt.savefig(f'video-images/knn-predictions/prediction-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()

            plt.figure()
            plt.imshow(np.reshape(temp_confidence, (50, 50)), vmin=min_limit,
                       vmax=max_limit, cmap='Reds')  # (1, 50, 50, 1)
            plt.colorbar()
            plt.savefig(f'video-images/knn-confidence/confidence-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()
        else:
            plt.figure()
            plt.imshow(np.zeros((50, 50)), vmin=min_limit,
                       vmax=max_limit, cmap='RdYlGn_r')  # (1, 50, 50, 1)
            plt.colorbar()
            plt.savefig(f'video-images/knn-predictions/prediction-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()

            plt.figure()
            plt.imshow(np.zeros((50, 50)), vmin=min_limit,
                       vmax=max_limit, cmap='Reds')  # (1, 50, 50, 1)
            plt.colorbar()
            plt.savefig(f'video-images/knn-confidence/confidence-{index}.png')
            plt.xlim([0, 50])
            plt.ylim([0, 50])
            plt.close()

    if sensor:
        sensor_data_past = sensor_data[np.where(
            (sensor_data[:, 0] <= i) & (sensor_data[:, 0] >= (i - 5)))]

        plt.figure()
        if len(sensor_data_past) == 0:
            plt.scatter(sensor_data_past[:, 1], sensor_data_past[:, 2])
        else:
            pz = np.array((sensor_data_past[:, 0] - np.min(sensor_data_past[:, 0])) / (
                np.max(sensor_data_past[:, 0]) - np.min(sensor_data_past[:, 0])))
            colors = np.array([[176 / 255, 10 / 255, 20 / 255, 1.]]
                              * (len(sensor_data_past)))
            colors[:, -1] = pz
            plt.scatter(
                sensor_data_past[:, 1], sensor_data_past[:, 2], color=colors, marker='.')
        # plt.show()

        plt.xlim([0, 50])
        plt.ylim([0, 50])

        plt.savefig(f'video-images/trajectories/trajectories-{index}.png')

        plt.close()

    if gt:

        data_filtered = ground_truth[np.where(ground_truth[:, 0] == i)]

        true = data_filtered[:, 3]

        plt.figure()
        plt.imshow(np.reshape(true, (50, 50)), vmin=min_limit,
                   vmax=max_limit, cmap='RdYlGn_r')  # (1, 50, 50, 1)
        plt.colorbar()
        plt.savefig(f'video-images/gt/gt-{index}.png')
        plt.close()
