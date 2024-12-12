import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod


class BaseEnvironment(gym.Env):
    """
    Base class for Gymnasium environments processing sensor data.

    This class abstracts common functionalities for environments that handle
    time-series sensor data, including resetting, stepping, and computing rewards.

    Attributes:
        data (pd.DataFrame): Loaded dataset containing sensor data.
        current_step (int): Current timestep within an episode.
        max_steps (int): Maximum number of steps per episode.
        data_path (str): Path to the dataset CSV file.
        ground_truth (float): The true value for the current step (to be predicted).
    """

    def __init__(self, **kwargs):
        """
        Initialize the base environment.

        Args:
            data_path (str): Path to the dataset CSV file.
            max_steps (int): Maximum number of steps per episode.
        """
        super(BaseEnvironment, self).__init__()
        # self.data_path = data_path
        # self.max_steps = max_steps

        # Placeholder attributes
        self.data: np.ndarray = None
        self.current_step = 0
        self.ground_truth = None

        self._initialize_attributes(**kwargs)
        self._load_data()
        self._pre_process_data()
        self._set_spaces()

    
    def _initialize_attributes(self, **kwargs):
        """
        Initialize attributes specific to the subclass.

        This method should be overridden by subclasses to define
        subclass-specific attributes before `_set_spaces` is called.
        """
        
        for key, value in kwargs.items():
            setattr(self, key, value)  # Dynamically set each attribute

        # Optionally, enforce required attributes
        required_attributes = ["data_path", "max_steps"]
        for attr in required_attributes:
            if not hasattr(self, attr):
                raise ValueError(f"Missing required attribute: {attr}")

    def _load_data(self):
        """
        Load the dataset from the specified path.

        Args:
            data_path (str): Path to the dataset file.

        Raises:
            ValueError: If the dataset is empty or invalid.
            FileNotFoundError: If the file does not exist.
        """
        try:
            self.data = pd.read_csv(self.data_path)
            if self.data.empty:
                raise ValueError(f"The dataset at {data_path} is empty.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {data_path}")
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

    @abstractmethod
    def _pre_process_data(self):
        """
        Pre-process the dataset to add necessary attributes or columns.

        This method should be overridden by subclasses to perform any necessary
        pre-processing steps on the dataset.
        """
        raise NotImplementedError(
            "`_pre_process_data` must be implemented in the subclass to define observation and action spaces ranges."
        )

    @abstractmethod
    def _set_spaces(self):
        """
        Define observation and action spaces.

        This method should be overridden by subclasses to specify
        the observation space and action space as required by the specific environment.
        """
        pass

    @abstractmethod
    def _compute_observation(self):
        pass

    @abstractmethod
    def _compute_ground_truth(self):
        pass

    @abstractmethod
    def _compute_reward(self, action, ground_truth):
        pass

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for reset.

        Returns:
            tuple: Initial observation and additional information.
        """
        super().reset(seed=seed)
        #self.current_step = 0

        #observation, self.ground_truth = (
        #    self._compute_observation(),
        #    self._compute_ground_truth(),
        #)

        #return observation, {"ground_truth": self.ground_truth}
        pass

    def step(self, action):
        """
        Take an action and move to the next step.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            tuple: Observation, reward, termination status, truncation status, and additional info.
        """
        reward = self._compute_reward(action, self.ground_truth)
        self.current_step += 1

        terminated = self._compute_terminated()
        if not terminated:
            observation, self.ground_truth = (
                self._compute_observation(),
                self._compute_ground_truth(),
            )
        else:
            observation = None

        return (
            observation,
            reward,
            terminated,
            False,
            {"ground_truth": self.ground_truth},
        )

    @abstractmethod
    def _compute_terminated(self):
        """
        Check if the current episode is over.

        Returns:
            bool: True if the episode has ended, False otherwise.
        """
        pass

    def render(self, mode="human"):
        """
        Render the environment's current state.

        Args:
            mode (str): The mode of rendering. Defaults to "human".
        """
        print(f"Step: {self.current_step}, Ground Truth: {self.ground_truth}")

    def close(self):
        """
        Cleanup resources before exiting.
        """
        pass
