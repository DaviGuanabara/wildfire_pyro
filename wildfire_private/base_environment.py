import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np


class SensorEnvironment(gym.Env):
    """
    Gymnasium environment for processing sensor data.

    Each episode corresponds to one sensor's lifetime, and the goal is to predict
    the variable `y` for the given observations `t`, `lat`, and `lon`.
    """

    def __init__(self, data_path, max_steps=50):
        super(SensorEnvironment, self).__init__()

        # Load the data
        self.data = pd.read_csv(data_path)
        self.sensors = self.data['lat'].unique()
        self.max_steps = max_steps

        # Observation space: [t, lat, lon]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        # Action space: Predicting `y` (continuous value)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        self.current_step = 0
        self.current_sensor = None
        self.current_data = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        """
        super().reset(seed=seed)

        # Randomly select a sensor for this episode
        self.current_sensor = np.random.choice(self.sensors)
        self.current_data = self.data[self.data['lat'] == self.current_sensor].reset_index(drop=True)
        self.current_step = 0

        # First observation
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        """
        Take an action and move to the next step.
        """
        # Get the true value of `y`
        y_desired = self.current_data.iloc[self.current_step]['y']

        # Calculate reward (negative mean squared error)
        predicted_y = action[0]
        reward = -np.square(predicted_y - y_desired)

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.current_data) or self.current_step >= self.max_steps

        # Get next observation or end episode
        if done:
            obs = None
        else:
            obs = self._get_obs()

        return obs, reward, done, False, {y_desired: y_desired}

    def _get_obs(self):
        """
        Get the current observation.
        """
        row = self.current_data.iloc[self.current_step]
        return np.array([row['t'], row['lat'], row['lon']], dtype=np.float32)

    def render(self, mode="human"):
        """
        Optional: Render the environment's state.
        """
        print(f"Step: {self.current_step}, Sensor: {self.current_sensor}")

    def close(self):
        """
        Cleanup resources.
        """
        pass
