from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from .predict_utils import predict_model
from gymnasium import spaces

class BaseTargetProvider(ABC):
    def __init__(self, target_info_key: str = "", observation_space: spaces = None, device: str = ""):
        self.target_info_key:str = target_info_key        
        self.observation_space:spaces = observation_space
        self.device:str = device
        

    @abstractmethod
    def get_target(self, info: Dict[str, Any]) -> torch.Tensor:
        pass


class InfoFieldTargetProvider(BaseTargetProvider):
    def __init__(self, target_info_key: str = "ground_truth"):
        super().__init__(target_info_key=target_info_key)

    def get_target(self, info: Dict[str, Any]) -> Any:
        if info is None:
            raise ValueError(
                "InfoFieldTargetProvider requires 'info' dictionary.")
        return info.get(self.target_info_key, None)


class TeacherTargetProvider(BaseTargetProvider):
    """
    Uses a pre-trained teacher network to generate targets.
    Expects 'teacher_observation' to be included in the info dictionary.

    Raises:
        ValueError: If observation is missing or mismatched in shape.
    """
    def __init__(
        self,
        teacher: torch.nn.Module,
        observation_space: spaces,
        device: str = "cpu",
    ):
        super().__init__(observation_space=observation_space, device=device)
        self.teacher = teacher.to(device)
        self.device = device

    def get_target(self, info: Dict[str, Any]) -> torch.Tensor:
        
        teacher_observation = info.get("teacher_observation", None)
        self._verify_observation_shape(teacher_observation)
            
        prediction, _ = predict_model(
            self.teacher,
            teacher_observation,
            device=self.device,
            input_shape=self.observation_space.shape,
        )

        return prediction
    
    def _verify_observation_shape(self, obs: np.ndarray) -> None:
        if self.observation_space is None:
            raise ValueError("Observation space must be provided.")
        if obs is None:
            raise ValueError("Observation must be provided.")
        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"[TeacherTargetProvider] Observation shape mismatch.\n"
                f"Expected: {self.observation_space.shape}, Got: {obs.shape}\n"
                f"Hint: Ensure teacher_observation is passed correctly from the environment."
            )
