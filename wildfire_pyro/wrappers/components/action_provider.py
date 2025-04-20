# wildfire_pyro/wrappers/components/action_source.py

import numpy as np
import torch
import torch.nn as nn


class BaseActionProvider:
    def __init__(self, provider: nn.Module, device: str):
        self.provider = provider
        self.device = device

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.provider(obs_tensor).cpu().numpy().squeeze(0)
        return action


class StudentActionProvider(BaseActionProvider):
    def __init__(self, student: nn.Module, device: str):
        super().__init__(student, device)


class TeacherActionProvider(BaseActionProvider):
    def __init__(self, teacher: nn.Module, device: str):
        super().__init__(teacher, device)
