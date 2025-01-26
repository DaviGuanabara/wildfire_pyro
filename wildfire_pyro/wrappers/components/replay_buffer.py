import torch
import numpy as np
from typing import Any, Dict, Tuple

from gymnasium import spaces
from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
import logging
from wildfire_pyro.environments.base_environment import BaseEnvironment

import torch
import numpy as np
from typing import Tuple, Optional


class ReplayBuffer:
    """
    Stores transitions (observation, action, ground_truth) for training.
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
        self.ground_truth = torch.zeros((max_size, 1), device=device)

        self.position = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        ground_truth: float,
    ):
        """
        Adds a transition to the buffer.

        Args:
            obs (np.ndarray): Current observation.
            action (np.ndarray): Action taken.
            ground_truth (float): Optimal action associated.
        """
        if self.position < self.max_size:
            self.observations[self.position] = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            )
            self.actions[self.position] = torch.tensor(
                action, device=self.device, dtype=torch.float32
            )
            self.ground_truth[self.position] = torch.tensor(
                ground_truth, device=self.device, dtype=torch.float32
            )
            self.position += 1

            if self.position == self.max_size:
                self.full = True
        else:
            raise RuntimeError(
                "ReplayBuffer is full. Call `reset` before adding more transitions."
            )

    def sample_batch(self) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns a batch of data from the buffer.

        Returns:
            Tuple containing observations, actions, and ground_truth.
        """
        if not self.full:
            raise ValueError("Buffer is not full yet.")
        # Return the entire buffer as the batch
        return (
            self.observations,
            self.actions,
            self.ground_truth,
        )

    def reset(self):
        """
        Resets the buffer by clearing all stored transitions.
        """
        self.position = 0
        self.full = False
        self.observations.zero_()
        self.actions.zero_()
        self.ground_truth.zero_()

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
