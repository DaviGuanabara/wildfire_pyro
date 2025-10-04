from abc import ABC, abstractmethod
from gymnasium import Env
from typing import Any, Tuple, Dict, Optional
import numpy as np

class BaseEnvironment(Env, ABC):
    """
    Abstract base class for environments, inheriting from Gymnasium's Env.
    Child classes must implement the abstract methods and properties.
    """

    def __init__(self):
        self._context_handlers = {}

    @abstractmethod
    def baseline(self, bootstrap_observations: np.ndarray) -> np.ndarray:
        """Returns a baseline prediction. Should be overridden if applicable.
        it should return:
        prediction"""
        return np.array([])

    def render(self, mode: str = "human") -> Any:
        """
        Renders the environment.

        Args:
            mode (str, optional): The mode to render with. Defaults to 'human'.

        Returns:
            Any: The rendered image or other output depending on the mode.
        """
        pass

    def close(self):
        """
        Performs any necessary cleanup.
        """
        pass

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Default implementation that sets the random seed.
        Subclasses should override this method to return actual observations.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional reset options.

        Returns:
            Tuple[Any, Dict[str, Any]]: Initial observation and info (empty by default).
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        return None, {}  # placeholder, subclasses devem sobrescrever


    @abstractmethod
    def step(self, action: Optional[Any] = None) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        Args:
            action (Any): An action provided by the agent.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]:
                - observation (Any): Agent's observation of the current environment.
                - reward (float): Amount of reward returned after previous action.
                - terminated (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated.
                - info (Dict[str, Any]): Diagnostic information.
        """
        pass

    

    @abstractmethod
    def get_bootstrap_observations(self, n_bootstrap: int, force_recompute: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def bootstrap_baseline(self) -> np.ndarray:
        return np.array([])
    
