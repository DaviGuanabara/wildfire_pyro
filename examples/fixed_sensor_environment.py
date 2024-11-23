from wildfire_private.wildfire_private.base_environment import BaseEnvironment as BaseEnvironment
import numpy as np
import pandas as pd
from gymnasium import spaces


class Fixed_Sensor_Environment(BaseEnvironment):
    def __init__(self, data_path, max_steps=50, n_neighbors_min=5, n_neighbors_max=10):
        super(Fixed_Sensor_Environment, self).__init__(data_path, max_steps=max_steps)
        self.n_neighbors_min = n_neighbors_min
        self.n_neighbors_max = n_neighbors_max
        self.initial_step = 2  # Começa no terceiro timestep
        self.current_step = self.initial_step
        self.episode_done = False

        # Pre-process the dataset to add `sensor_id` and compute ranges
        self._pre_process_data()
        self._set_observation_spaces()

    def _pre_process_data(self):
        """
        Add a unique `sensor_id` for each unique combination of latitude and longitude.
        Compute min and max values for lat, lon, t, and y to populate the observation space.
        """
        # Add sensor_id
        self.data['sensor_id'] = self.data.groupby(['lat', 'lon']).ngroup()

        # Compute min and max for each feature in the observation space
        self.lat_min = self.data['lat'].min()
        self.lat_max = self.data['lat'].max()
        self.lon_min = self.data['lon'].min()
        self.lon_max = self.data['lon'].max()
        self.t_min = self.data['t'].min()
        self.t_max = self.data['t'].max()
        self.y_min = self.data['y'].min()
        self.y_max = self.data['y'].max()

    def _set_observation_spaces(self):
        """
        Define observation and action spaces.
        """
        lat_range = self.lat_max - self.lat_min
        lon_range = self.lon_max - self.lon_min
        t_range = 3 * np.pi
        y_range = self.y_max - self.y_min

        # Set the observation space using the computed min and max values
        self.observation_space = spaces.Box(
            low=np.array([
                -lat_range,  # Δlat (relative latitude)
                -lon_range,  # Δlon (relative longitude)
                -t_range,    # Δt (time difference is within 3π)
                self.y_min   # y (neighboring sensor value)
            ]).repeat(self.n_neighbors_max).reshape(4, self.n_neighbors_max),
            high=np.array([
                lat_range,  # Δlat (relative latitude)
                lon_range,  # Δlon (relative longitude)
                0,          # Δt (no time difference for neighbors in the past)
                self.y_max  # y (neighboring sensor value)
            ]).repeat(self.n_neighbors_max).reshape(4, self.n_neighbors_max),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        """
        super().reset(seed=seed)
        self.current_step = self.initial_step
        self.episode_done = False

        # Randomly select a sensor for the episode
        self.target_sensor = np.random.choice(self.data['sensor_id'].unique())

        # Compute the initial observation and ground truth
        observation, self.ground_truth = self._compute_observation()
        return observation, {"ground_truth": self.ground_truth}

    def _compute_observation(self):
        """
        Compute the observation for the current timestep.
        """
        sensor_data = self.data[self.data['sensor_id'] == self.target_sensor]
        if self.current_step >= len(sensor_data):
            raise IndexError("No more data for the selected sensor.")

        # Get current timestep data
        current_time = sensor_data.iloc[self.current_step]['t']
        target_row = sensor_data.iloc[self.current_step]

        # Select neighbors within the last 3π time range
        time_range_start = max(current_time - 3 * np.pi, 0)
        neighbors = sensor_data[
            (sensor_data['t'] >= time_range_start) & (sensor_data['t'] < current_time)
        ]

        # Sample a random number of neighbors
        n_neighbors = np.random.randint(self.n_neighbors_min, self.n_neighbors_max + 1)
        neighbors = neighbors.sample(min(len(neighbors), n_neighbors)) if not neighbors.empty else pd.DataFrame()

        # Create observation matrix and mask
        u_matrix = np.zeros((4, self.n_neighbors_max), dtype=np.float32)
        mask = np.zeros(self.n_neighbors_max, dtype=bool)

        for i, (_, neighbor) in enumerate(neighbors.iterrows()):
            if i >= self.n_neighbors_max:
                break
            u_matrix[:, i] = [
                neighbor['lat'] - target_row['lat'],
                neighbor['lon'] - target_row['lon'],
                neighbor['t'] - current_time,
                neighbor['y']
            ]
            mask[i] = True

        # Ground truth for the target timestep
        ground_truth = target_row['y']

        return (u_matrix, mask), ground_truth

    def _compute_reward(self, action, ground_truth):
        """
        Compute the reward as negative Mean Squared Error (MSE).
        """
        predicted = action[0]
        return -np.square(predicted - ground_truth)

    def render(self, mode="human"):
        """
        Render the current state of the environment.
        """
        print(f"Step {self.current_step}/{self.max_steps}, "
              f"Target Sensor: {self.target_sensor}, "
              f"Ground Truth: {self.ground_truth:.3f}")

    def close(self):
        """
        Clean up resources.
        """
        pass
