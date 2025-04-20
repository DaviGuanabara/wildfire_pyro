from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from .predict_utils import predict_model


class BaseTargetProvider(ABC):
    def __init__(self, target_info_key: str = "ground_truth", input_shape: Any = None):
        self.target_info_key = target_info_key
        self.input_shape = input_shape

    @abstractmethod
    def get_target(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        pass


class InfoFieldTargetProvider(BaseTargetProvider):
    def __init__(self, target_info_key: str = "ground_truth"):
        super().__init__(target_info_key=target_info_key)

    def get_target(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> Any:
        if info is None:
            raise ValueError(
                "InfoFieldTargetProvider requires 'info' dictionary.")
        return info.get(self.target_info_key, None)


class TeacherTargetProvider(BaseTargetProvider):
    def __init__(
        self,
        teacher: torch.nn.Module,
        target_info_key: str = "teacher_prediction",
        input_shape: Any = None,
        device: str = "cpu",
    ):
        super().__init__(target_info_key=target_info_key, input_shape=input_shape)
        self.teacher = teacher.to(device)
        self.device = device

    def get_target(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        prediction, _ = predict_model(
            self.teacher,
            obs,
            device=self.device,
            input_shape=self.input_shape,
        )

        return prediction
