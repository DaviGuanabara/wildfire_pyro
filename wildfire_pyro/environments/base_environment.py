from abc import ABC, abstractmethod
from gymnasium import Env, spaces
from typing import Any, Tuple, Dict


class BaseEnvironment(Env, ABC):
    """
    Abstract base class for environments, inheriting from Gymnasium's Env.
    Child classes must implement the abstract methods and properties.
    """

    @property
    @abstractmethod
    def observation_space(self) -> spaces.Space:
        """
        The observation space of the environment.

        Returns:
            spaces.Space: The observation space.
        """
        pass

    @property
    @abstractmethod
    def action_space(self) -> spaces.Space:
        """
        The action space of the environment.

        Returns:
            spaces.Space: The action space.
        """
        pass

    @abstractmethod
    def reset(
        self, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Resets the environment to an initial state and returns the initial observation.

        Args:
            seed (int, optional): The seed for the environment's random number generator.
            options (dict, optional): Additional options for resetting the environment.

        Returns:
            Tuple[Any, Dict[str, Any]]: The initial observation and additional info.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Executes one time step within the environment.

        Args:
            action (Any): An action provided by the agent.

        Returns:
            Tuple[Any, float, bool, bool, Dict[str, Any]]:
                - observation (Any): Agent's observation of the current environment.
                - reward (float): Amount of reward returned after previous action.
                - done (bool): Whether the episode has ended.
                - truncated (bool): Whether the episode was truncated.
                - info (Dict[str, Any]): Diagnostic information.
        """
        pass

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
