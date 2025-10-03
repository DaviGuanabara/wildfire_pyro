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

    def __init__(self, max_size: int, device: str = "cpu"):
        self.device = device
        self.max_size = max_size
        self.position = 0
        self.full = False

        # Armazenamento genérico
        self.observations = [None] * max_size
        self.actions = [None] * max_size
        self.targets = [None] * max_size

    def add(self, obs: Any, action: Any, target: Any):
        """
        Adiciona uma transição ao buffer.
        Obs pode ser tensor, numpy, dict, etc.
        """
        idx = self.position % self.max_size

        self.observations[idx] = self._to_tensor(obs) #type: ignore
        self.actions[idx] = self._to_tensor(action) #type: ignore
        self.targets[idx] = self._to_tensor(target) #type: ignore

        self.position += 1
        if self.position >= self.max_size:
            self.full = True

    def sample_batch(self, batch_size: int):
        buffer_size = self.max_size if self.full else self.position
        if buffer_size < batch_size:
            raise ValueError(f"Not enough samples: {buffer_size} < {batch_size}")

        indices = np.random.choice(buffer_size, batch_size, replace=False)

        obs_batch = [self.observations[i] for i in indices]
        act_batch = [self.actions[i] for i in indices]
        tgt_batch = [self.targets[i] for i in indices]

        # 🔹 Se observação for tensor
        if isinstance(obs_batch[0], torch.Tensor):
            obs_batch = torch.stack(obs_batch)

        # 🔹 Se observação for dict
        elif isinstance(obs_batch[0], dict):
            obs_batch = {k: torch.stack([d[k] for d in obs_batch])
                        for k in obs_batch[0]}

        return (
            obs_batch,
            torch.stack(act_batch),
            torch.stack(tgt_batch),
        )



    def reset(self):
        """Limpa o buffer."""
        self.position = 0
        self.full = False
        self.observations: List[Optional[torch.Tensor]] = [None] * self.max_size
        self.actions: List[Optional[torch.Tensor]] = [None] * self.max_size
        self.targets: List[Optional[torch.Tensor]] = [None] * self.max_size

    def is_full(self) -> bool:
        return self.full

    def size(self) -> int:
        return self.max_size if self.full else self.position

    def pop_oldest(self):
        """Remove a transição mais antiga."""
        if self.position == 0 and not self.full:
            raise ValueError(
                "Buffer is empty. Cannot remove oldest transition.")

        # Desloca elementos (O(n), mas simples)
        self.observations.pop(0)
        self.actions.pop(0)
        self.targets.pop(0)

        # Adiciona espaço vazio no fim
        self.observations.append(None)
        self.actions.append(None)
        self.targets.append(None)

        if not self.full:
            self.position -= 1
        elif self.position == self.max_size:
            self.full = False

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
