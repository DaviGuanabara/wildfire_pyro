import torch
import numpy as np
from typing import Any, Dict, List, Tuple

from gymnasium import spaces
from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
import logging
from wildfire_pyro.environments.base_environment import BaseEnvironment

import torch
import numpy as np
from typing import Tuple, Optional, Union


import torch
import numpy as np
from typing import Any, Tuple


class ReplayBuffer:
    """
    Generic Replay Buffer que armazena (obs, action, target) sem assumir
    formato fixo do espaço de observação. Usa listas para máxima flexibilidade.
    """

    def __init__(self, max_size: int, seed: int, device: str = "cpu"):
        self.device = device
        self.max_size = max_size

        self.position = 0

        # Armazenamento genérico
        self.observations = [None] * max_size
        self.actions = [None] * max_size
        self.targets = [None] * max_size
        self.seed = seed
        self.reset(seed)

    def add(self, obs: Any, action: Any, target: Any):
        """
        Add a new observation into the buffer.
        basically, it is ciclic, writing over old data when full.
        First In First Out.
        """
        idx = self.position % self.max_size

        self.observations[idx] = self._to_tensor(obs)  # type: ignore
        self.actions[idx] = self._to_tensor(action)  # type: ignore
        self.targets[idx] = self._to_tensor(target)  # type: ignore

        self.position += 1

        # Limits integer overflow for long training runs.
        if self.position >= 2 * self.max_size:
            self.position = self.max_size

    def sample_batch(self, batch_size: int) -> Tuple[Any, torch.Tensor]:
        buffer_size = self.n_occupied_slots()

        if buffer_size < batch_size:
            raise ValueError(f"Not enough samples: {buffer_size} < {batch_size}")

        if not hasattr(self, "rng"):
            raise ValueError(
                "[REPLAY BUFFER] Random number generator not initialized. Call reset(seed) before using this method."
            )

        indices = self.rng.choice(buffer_size, batch_size, replace=False)

        obs_batch = [self.observations[i] for i in indices]
        tgt_batch = [self.targets[i] for i in indices]

        # 🔹 Se observação for tensor
        if isinstance(obs_batch[0], torch.Tensor):
            obs_batch = torch.stack(obs_batch)

        # 🔹 Se observação for dict
        elif isinstance(obs_batch[0], dict):
            obs_batch = {
                k: torch.stack([d[k] for d in obs_batch]) for k in obs_batch[0]
            }

        return (
            obs_batch,
            torch.stack(tgt_batch),
        )

    def reset(self, seed: Optional[int] = None):
        """
        Clear the buffer.
        Currently, is only called by itself, in __init__.
        If we want to call it from outside, we should be careful, because it will reset the seed
        """

        self.position = 0
        self.observations: List[Optional[torch.Tensor]] = [None] * self.max_size
        self.actions: List[Optional[torch.Tensor]] = [None] * self.max_size
        self.targets: List[Optional[torch.Tensor]] = [None] * self.max_size

        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

    def is_full(self) -> bool:
        """
        Check if the buffer is full.
        position says where it is writing, so if position >= max_size,
        it means it has already written max_size elements, and is now overwriting old data.
        """
        return self.position >= self.max_size

    def n_occupied_slots(self) -> int:
        return self.max_size if self.is_full() else self.position

    def n_empty_slots(self) -> int:
        return self.max_size - self.n_occupied_slots()

    # -----------------------------
    # Helpers
    # -----------------------------
    def _to_tensor(self, x: Any):
        """Converte entradas para tensor (ou dict de tensores)."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, dict):
            # mantém dict, mas garante que cada valor seja tensor
            return {k: self._to_tensor(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple, np.ndarray, float, int)):
            return torch.as_tensor(x, device=self.device, dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported type in ReplayBuffer: {type(x)}")
