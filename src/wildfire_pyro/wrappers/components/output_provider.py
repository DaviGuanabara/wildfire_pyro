# wildfire_pyro/wrappers/components/output_provider.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from gymnasium import spaces
from .predict_utils import predict_model

class BaseOutputProvider(ABC):
    """
    General interface for anything that produces an output
    given an observation or info dict.
    """

    @abstractmethod
    def get_output(self, obs: np.ndarray) -> np.ndarray:
        pass


class LabelProvider(BaseOutputProvider):
    """
    Reads the output directly from the info dictionary.
    Useful for ground truth values.
    """
    def __init__(self, info_key: str = "ground_truth"):
        super().__init__()
        self.info_key = info_key

    def get_output(self, obs: np.ndarray ) -> np.ndarray:
        return obs


class PredictionProvider(BaseOutputProvider):
    """
    Uses a neural network to produce outputs from observations.
    """
    def __init__(self, network: torch.nn.Module, observation_space: spaces.Space, device: str = "cpu"):
        super().__init__()
        self.network = network.to(device)
        self.observation_space = observation_space
        self.device = device

    def get_output(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns:
        Tuple[np.ndarray, dict]: 
            - Model prediction(s), shape depends on input format.
            - Info dictionary (currently empty, but expandable).
        """
        if obs is None:
            raise ValueError("PredictionProvider requires an observation.")
        prediction, _ = predict_model(
            self.network,
            obs,
            device=self.device,
            observation_space=self.observation_space,
        )
        return prediction

    def get_nn(self) -> torch.nn.Module:
        return self.network


class TeacherPredictionProvider(PredictionProvider):
    """
    Uses a neural network to produce outputs from teacher observations.
    """
    def __init__(self, network: torch.nn.Module, observation_space: spaces.Space, device: str = "cpu"):
        super().__init__(network, observation_space, device)
   

    def get_output(self, obs: np.ndarray) -> np.ndarray:
        """
        Uses student observations to get model predictions.
        Collects observation from info[student_observation].
        Returns:
            torch.Tensor: Model prediction(s), shape depends on input format.
        """
        return super().get_output(obs)

class StudentPredictionProvider(PredictionProvider):
    """
    Uses a neural network to produce outputs from student observations.
    """
    def __init__(self, network: torch.nn.Module, observation_space: spaces.Space, device: str = "cpu"):
        super().__init__(network, observation_space, device)

    def get_output(self, obs: np.ndarray) -> np.ndarray:
        """
        Uses student observations to get model predictions.
        Collects observation from info[student_observation].
        Returns:
            torch.Tensor: Model prediction(s), shape depends on input format.
        """
        return super().get_output(obs)

