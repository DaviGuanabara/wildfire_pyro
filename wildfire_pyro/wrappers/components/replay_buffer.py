import torch
import numpy as np
from typing import Any, Dict, Tuple

from gymnasium import spaces
from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
import logging
from wildfire_pyro.environments.base_environment import BaseEnvironment

import torch
import numpy as np
from typing import Tuple, Optional, Union


class ReplayBuffer:
    """
    Stores transitions (observation, action, target) for training.
    Implements a fixed-size buffer equal to batch_size.
    """

    def __init__(
        self,
        max_size: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: str = "cpu",
    ):
        """
        Initializes the ReplayBuffer.

        Args:
            max_size (int): Number of transitions to store (typically equal to batch_size).
            observation_shape (Tuple[int, ...]): Shape of the observation space.
            action_shape (Tuple[int, ...]): Shape of the action space.
            device (str, optional): Device to store tensors. Defaults to "cpu".
        """
        self.device = device
        self.max_size = max_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        # Initialize tensors to store transitions
        self.observations = torch.zeros(
            (max_size,) + observation_shape, device=device)
        self.actions = torch.zeros((max_size,) + action_shape, device=device)
        self.target = torch.zeros((max_size, 1), device=device)

        self.position = 0
        self.full = False

    def add(self, obs: np.ndarray, action: np.ndarray, target: Union[float, torch.Tensor]):
        """
        Adds a transition to the buffer, replacing the oldest transition if the buffer is full.

        Args:
            obs (np.ndarray): Current observation.
            action (np.ndarray): Action taken.
            target (float): Optimal action associated.
        """
        # Use circular buffer logic: Overwrite the oldest data instead of throwing an error
        idx = self.position % self.max_size  # Circular index

        self.observations[idx] = torch.tensor(
            obs, device=self.device, dtype=torch.float32
        )
        self.actions[idx] = torch.tensor(
            action, device=self.device, dtype=torch.float32
        )
        self.target[idx] = torch.tensor(
            target, device=self.device, dtype=torch.float32
        )

        # Update position and tracking
        self.position += 1
        if self.position >= self.max_size:
            self.full = True  # Mark buffer as full after first complete cycle

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns a randomly sampled batch from the buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            Tuple containing observations, actions, and target.
        """
        # Ensure there are enough samples to draw a batch
        buffer_size = self.max_size if self.full else self.position

        if buffer_size < batch_size:
            raise ValueError(
                f"Not enough samples in buffer. Requested: {batch_size}, Available: {buffer_size}")

        # Select `batch_size` random indices while ensuring alignment between observation, actions, and target
        indices = np.random.choice(buffer_size, batch_size, replace=False)

        return (
            self.observations[indices],
            self.actions[indices],
            self.target[indices],
        )

    def reset(self):
        """
        Resets the buffer by clearing all stored transitions.
        """
        self.position = 0
        self.full = False
        self.observations.zero_()
        self.actions.zero_()
        self.target.zero_()

    def is_full(self) -> bool:
        """
        Checks if the buffer is full.

        Returns:
            bool: True if buffer is full, False otherwise.
        """
        return self.full

    def size(self) -> int:
        """
        Returns the current number of transitions stored in the buffer.

        Returns:
            int: Number of transitions stored.
        """
        return self.max_size if self.full else self.position

    def pop_oldest(self):
        """
        Removes the oldest transition from the buffer by shifting all elements to the left.
        """
        if self.position == 0 and not self.full:
            raise ValueError("Buffer is empty. Cannot remove oldest transition.")

        # Shift all elements to the left
        self.observations[:-1] = self.observations[1:].clone()
        self.actions[:-1] = self.actions[1:].clone()
        self.target[:-1] = self.target[1:].clone()


        # Clear the last position
        self.observations[-1].zero_()
        self.actions[-1].zero_()
        self.target[-1].zero_()

        # Adjust position tracking
        if not self.full:
            self.position -= 1
        elif self.position == self.max_size - 1:
            self.full = False
