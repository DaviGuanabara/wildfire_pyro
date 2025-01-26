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

from wildfire_pyro.wrappers.components.replay_buffer import ReplayBuffer


class EnvDataCollector:
    """
    Interacts with the environment and collects data for the ReplayBuffer.
    """

    def __init__(
        self, environment: BaseEnvironment, buffer: ReplayBuffer, device: str = "cpu"
    ):
        """
        Initializes the EnvDataCollector.

        Args:
            environment (Any): Gymnasium environment instance.
            buffer (ReplayBuffer): Instance of ReplayBuffer to store data.
            device (str, optional): Device to store tensors. Defaults to "cpu".
        """
        self.environment: BaseEnvironment = environment
        self.buffer = buffer
        self.device = device

    def collect_rollouts(self, neural_network: torch.nn.Module, n_rollout_steps: int):
        """
        Collects rollouts from the environment and stores them in the buffer.

        Args:
            neural_network (torch.nn.Module): Neural network for action prediction.
            n_rollout_steps (int): Number of steps to collect.
        """
        obs, info = self.environment.reset()

        for step in range(n_rollout_steps):

            if self.buffer.is_full():
                self.buffer.reset()

            with torch.no_grad():

                # (1, output_dim)
                obs_tensor = torch.tensor(
                    obs, device=self.device, dtype=torch.float32
                ).unsqueeze(0)

                # TODO: NÃO PRECISO PEGAR O ACTION. ISSO PODE ATÉ DEIXAR O CÓDIGO MAIS RÁPIDO
                # MAS COMO SERIA A VALIDAÇÃO ?
                y_pred: torch.Tensor = neural_network(obs_tensor)
                action: np.ndarray = y_pred.cpu().numpy().squeeze(0)  # (output_dim,)

            ground_truth: Optional[float] = info.get("ground_truth", None)

            if ground_truth is None:
                print("[Warning] Missing ground_truth. Ending rollout.")
                break

            self.buffer.add(obs, action, ground_truth)

            obs, reward, terminated, truncated, info = self.environment.step(action)

            if terminated or truncated:
                # print("environment reseted")
                obs, info = self.environment.reset()
