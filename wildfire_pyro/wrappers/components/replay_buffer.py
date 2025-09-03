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

    def add(self, obs: np.ndarray, action: np.ndarray, target: np.ndarray):
        """
        Adds a transition to the buffer, replacing the oldest transition if the buffer is full.

        Args:
            obs (np.ndarray): Current observation.
            action (np.ndarray): Action taken.
            target (np.ndarray): Optimal action associated.
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


class DictReplayBuffer:
    def __init__(self, max_size: int, observation_space: spaces.Dict,
                 action_shape: Tuple[int, ...], device: str = "cpu"):
        self.device = device
        self.max_size = max_size
        self.position = 0
        self.full = False

        # 🔹 Inicializa tensores para cada chave
        self.observations: Dict[str, torch.Tensor] = {
            key: torch.zeros((max_size,) + space.shape, device=device)
            for key, space in observation_space.spaces.items()
        }
        self.actions = torch.zeros((max_size,) + action_shape, device=device)
        # <- mesma forma das actions
        self.target = torch.zeros((max_size,) + action_shape, device=device)

    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray, target: np.ndarray):
        """Adiciona uma transição ao buffer (sobrescreve quando cheio)."""
        idx = self.position % self.max_size

        # 🔹 Observações por chave
        for key, value in obs.items():
            t = torch.as_tensor(value, device=self.device, dtype=torch.float32)

            expected_shape = self.observations[key].shape[1:]  # sem batch

            # Caso 1: shape já está correto
            if t.shape == expected_shape:
                pass

            # Caso 2: veio com batch extra (1, *expected)
            elif t.shape[0] == 1 and t.shape[1:] == expected_shape:
                t = t.squeeze(0)

            # Caso 3: erro real
            else:
                raise ValueError(
                    f"[DictReplayBuffer] Chave '{key}' com shape {t.shape}, "
                    f"esperado {expected_shape} ou (1, {expected_shape})"
                )

            self.observations[key][idx] = t

        # 🔹 Ações
        act = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if act.ndim == len(self.actions.shape) - 1 and act.shape == self.actions.shape[1:]:
            pass  # já correto
        elif act.ndim == len(self.actions.shape) and act.shape[0] == 1:
            act = act.squeeze(0)
        self.actions[idx] = act

        # 🔹 Targets
        tgt = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        if tgt.ndim == len(self.target.shape) - 1 and tgt.shape == self.target.shape[1:]:
            pass
        elif tgt.ndim == len(self.target.shape) and tgt.shape[0] == 1:
            tgt = tgt.squeeze(0)
        self.target[idx] = tgt

        # Atualiza posição
        self.position += 1
        if self.position >= self.max_size:
            self.full = True




    def sample_batch(
        self, batch_size: int
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Retorna um batch aleatório (obs_dict, actions, target)."""
        buffer_size = self.max_size if self.full else self.position

        if buffer_size < batch_size:
            raise ValueError(
                f"Not enough samples in buffer. Requested: {batch_size}, Available: {buffer_size}"
            )

        indices = np.random.choice(buffer_size, batch_size, replace=False)

        obs_batch = {key: tensor[indices]
                     for key, tensor in self.observations.items()}
        return obs_batch, self.actions[indices], self.target[indices]

    def reset(self):
        """Limpa o buffer."""
        self.position = 0
        self.full = False
        for key, tensor in self.observations.items():
            tensor.zero_()
        self.actions.zero_()
        self.target.zero_()

    def is_full(self) -> bool:
        return self.full

    def size(self) -> int:
        return self.max_size if self.full else self.position

    def pop_oldest(self):
        """Remove a transição mais antiga, shiftando os dados."""
        if self.position == 0 and not self.full:
            raise ValueError(
                "Buffer is empty. Cannot remove oldest transition.")

        for key in self.observations.keys():
            self.observations[key][:-1] = self.observations[key][1:].clone()
            self.observations[key][-1].zero_()

        self.actions[:-1] = self.actions[1:].clone()
        self.actions[-1].zero_()

        self.target[:-1] = self.target[1:].clone()
        self.target[-1].zero_()

        if not self.full:
            self.position -= 1
        elif self.position == self.max_size - 1:
            self.full = False
