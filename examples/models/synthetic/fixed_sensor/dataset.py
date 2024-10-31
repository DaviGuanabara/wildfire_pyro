import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PointNeighborhood(Dataset):
    def __init__(self, 
                 data, 
                 min_neighbors=5, 
                 max_neighbors=30,
                 train=True,
                 random_noise=False,
                 noise_scale=0.01,
                 hidden=16,
                 normalize_time_difference=np.pi):


        # self.data.shape: (data_points, features)
        # example train_data: data shape: (2316, 4)
        self.data = data

        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        
        if normalize_time_difference != None:
            self.normalize_time_difference = normalize_time_difference
        else:
            self.normalize_time_difference = 1.0

        self.max_timestep_difference = normalize_time_difference

        self.train = train
        self.random_noise = random_noise
        self.noise_scale = noise_scale
        if self.train and self.random_noise:
            print(f"[INFO] applying normal random noise of std dev {self.noise_scale} to input")

        self.hidden = hidden

        self.training_point_indices = []
        # generate list of trainining point indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        n_neighbors = 0
        # check if there are enought points in the past, else sample a new point at random
        while n_neighbors < self.min_neighbors:

            data_point = self.data[index]
            
            t_i = data_point[0]

            # select only points samples in the past of the select point
            older_points = self.data[np.where((self.data[:, 0] < t_i) & (np.abs(self.data[:,0]-t_i) <= self.max_timestep_difference))]

            n_neighbors = len(older_points)

            if n_neighbors >= self.max_neighbors:
                # sample number of neighbors of point i
                neighbors = np.random.randint(low=self.min_neighbors, high=self.max_neighbors, size=1)
            elif n_neighbors >= self.min_neighbors and n_neighbors < self.max_neighbors:
                neighbors = n_neighbors
            else:
                #print(f'loop: {index}')
                index += 1

        # get point i
        point_i = np.expand_dims(data_point[:3], axis=0)
        # shape (1, 3)

        selection = np.random.randint(0, len(older_points), neighbors)

        point_j = older_points[selection]

        # calculate the features vectors for each neighbor j
        if self.train and self.random_noise:
            u_j = np.array([[(point_j[:,0]-point_i[0,0]/self.normalize_time_difference, 
                             (point_j[:,1]-point_i[0,1]+np.random.normal(0.0, self.noise_scale)),
                             (point_j[:,2]-point_i[0,2]+np.random.normal(0.0, self.noise_scale))),
                             point_j[:,3]]])
        else:
            u_j = np.array([[(point_j[:,0]-point_i[0,0])/self.normalize_time_difference, 
                             (point_j[:,1]-point_i[0,1]),
                             (point_j[:,2]-point_i[0,2]),
                             point_j[:,3]]])
        # shape: (1, 4, neighbors)
        u_j_t, mask = self.create_data_point(u_j, neighbors)

        # label of point i
        y_i = data_point[3].reshape(1,)

        return {'x_data': u_j_t, 
                'y_target': torch.tensor(y_i), 
                'mask': mask}

    def create_data_point(self, u_j, neighbors):

        # transpose the last dimensions of our matrix 
        # to the format our NN will accept
        u_j_t = np.moveaxis(u_j, -1, -2)
        # shape: (1, neighbors, 4)

        # remove the batch dimension of the data, it will be
        # added back by the dataloader using the batch dimension provided
        u_j_t = torch.squeeze(torch.tensor(u_j_t))

        # create mask to differentiate neighbors from padding
        mask = None

        # create a matrix of ones with the number of neighbors that point have
        ones = torch.ones(int(neighbors), self.hidden)

        if self.max_neighbors > neighbors:
            # if the point need padding because there is another point with more neighbors
            # add corresponding vectors of zeros, so this point has the same matrix dimension
            # as the point with most number of neighbors

            mask = torch.cat((ones, torch.zeros((self.max_neighbors-int(neighbors)), self.hidden)))
            u_j_t = torch.cat((u_j_t, torch.zeros((self.max_neighbors-int(neighbors)), 4)))

        return u_j_t, mask

    def generate_test_point(self, lat, lon, t):

        n_neighbors = 0
        # check if there are enought points in the past, else sample a new point at random
        while n_neighbors < self.min_neighbors:

            data_point = np.array([t, lat, lon])
            
            t_i = data_point[0]

            # select only points samples in the past of the select point
            older_points = self.data[np.where((self.data[:, 0] < t_i) & (np.abs(self.data[:,0]-t_i) <= self.max_timestep_difference))]

            n_neighbors = len(older_points)

            if n_neighbors >= self.max_neighbors:
                # sample number of neighbors of point i
                neighbors = np.random.randint(low=self.min_neighbors, high=self.max_neighbors, size=1)
            elif n_neighbors >= self.min_neighbors and n_neighbors < self.max_neighbors:
                neighbors = n_neighbors
            else:
                print(f'[error] insuficient number of neighbors')
                break

        # get point i
        point_i = np.expand_dims(data_point[:3], axis=0)
        # shape (1, 3)

        selection = np.random.randint(0, len(older_points), neighbors)

        point_j = older_points[selection]

        # calculate the features vectors for each neighbor j
        if self.train and self.random_noise:
            u_j = np.array([[(point_j[:,0]-point_i[0,0]/self.normalize_time_difference, 
                             (point_j[:,1]-point_i[0,1]+np.random.normal(0.0, self.noise_scale)),
                             (point_j[:,2]-point_i[0,2]+np.random.normal(0.0, self.noise_scale))),
                             point_j[:,3]]])
        else:
            u_j = np.array([[(point_j[:,0]-point_i[0,0])/self.normalize_time_difference, 
                             (point_j[:,1]-point_i[0,1]),
                             (point_j[:,2]-point_i[0,2]),
                             point_j[:,3]]])
        # shape: (1, 4, neighbors)
        u_j_t, mask = self.create_data_point(u_j, neighbors)

        return {'x_data': u_j_t, 
                'mask': mask}
        
    def get_num_batches(self, batch_size):

        return len(self) // batch_size

def generate_batches(dataset, 
                     batch_size, 
                     shuffle=True, 
                     drop_last=True, 
                     device="cpu"):

    dataloader = DataLoader(dataset=dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name,_ in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict

def read_data(path):

    df = pd.read_csv(path)
    data_np = df.to_numpy()

    print(f'data shape: {data_np.shape}')

    return data_np

