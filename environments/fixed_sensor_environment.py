from wildfire_pyro.environments.components.sensor_manager import SensorManager
from wildfire_pyro.environments.base_environment import BaseEnvironment

import numpy as np
import pandas as pd
from gymnasium import spaces

import logging

logging.basicConfig(level=logging.INFO)


from gymnasium import spaces


class FixedSensorEnvironment():
    def __init__(self, data_path, max_steps=50, n_neighbors_min=5, n_neighbors_max=10):
        """
        Initialize the Fixed Sensor Environment.

        Args:
            data_path (str): Path to the dataset.
            max_steps (int): Maximum steps per episode.
            n_neighbors_min (int): Minimum number of neighbors.
            n_neighbors_max (int): Maximum number of neighbors.
        """

        self.max_steps = max_steps
        self.n_neighbors_min = n_neighbors_min
        self.n_neighbors_max = n_neighbors_max
        self.data_path = data_path
        self.current_step = 0
        self.ground_truth = None

        self._pre_process_data()
        self.sensor_manager: SensorManager = SensorManager(data_path)

        self._set_spaces()

    def _pre_process_data(self):
        """
        Pre-process the dataset (example implementation).
        """
        self.data = pd.read_csv(self.data_path)
        self.data["sensor_id"] = self.data.groupby(["lat", "lon"]).ngroup()

    def _set_spaces(self):
        """
        Define observation and action spaces.
        """
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4, self.n_neighbors_max), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        """

        self.current_step = 0

        self.sensor_manager.update_sensor()
        observation = self._compute_observation()
        self.ground_truth = self._compute_ground_truth()
        return observation, {"ground_truth": self.ground_truth}

    def step(self, action):
        """
        Execute a step in the environment.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        try:
            observation = self._compute_observation()
            reward = self._compute_reward(action, self.ground_truth)
            self.ground_truth = self._compute_ground_truth()
            terminated = self._compute_terminated()
        except IndexError:
            terminated = True
            observation = None
            reward = 0.0

        return observation, reward, terminated, False, {"ground_truth": self.ground_truth}

    def _compute_reward(self, action, ground_truth):
        """
        Compute the reward for the current timestep.
        """
        return -(action - ground_truth)

    def _compute_observation(self):
        """
        Compute the observation for the current timestep.
        """
        target_row = self.sensor_manager.get_reading()
        neighbors = self.sensor_manager.get_neighbors(
            self.current_step,
            time_window=3 * np.pi,
            n_neighbors_min=self.n_neighbors_min,
            n_neighbors_max=self.n_neighbors_max,
        )
        u_matrix, mask = self._compute_u_matrix_and_mask(neighbors, target_row)
        return (u_matrix, mask)

    def _compute_ground_truth(self):
        """
        Compute the ground truth value for the current timestep.
        """
        target_row = self.sensor_manager.get_reading()
        return target_row["y"]

    def _compute_u_matrix_and_mask(self, neighbors, target_row):
        """
        Compute the observation matrix (u_matrix) and mask.
        """
        u_matrix = np.zeros((4, self.n_neighbors_max), dtype=np.float32)
        mask = np.zeros(self.n_neighbors_max, dtype=bool)

        for i, (_, neighbor) in enumerate(neighbors.iterrows()):
            if i >= self.n_neighbors_max:
                break
            u_matrix[:, i] = [
                neighbor["lat"] - target_row["lat"],
                neighbor["lon"] - target_row["lon"],
                neighbor["t"] - target_row["t"],
                neighbor["y"],
            ]
            mask[i] = True

        return u_matrix, mask

    def _compute_terminated(self):
        """
        Check if the episode should terminate.
        """
        if self.current_step >= self.max_steps:
            return True
        return False

    def _pre_process_data(self):
        """
        Pre-process the dataset.
        """
        pass
    
    def close(self):
        """
        Close the environment.
        """
        pass
