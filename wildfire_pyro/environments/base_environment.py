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

    def receive_context(self, context: dict):
        """
        Receives contextual data from external components such as callbacks,
        and dispatches the data to the appropriate handler registered in
        `self._context_handlers`.

        This method follows the Hollywood Principle: "Don't call us, we'll call you".
        In this architecture, external components (e.g., EvalCallback) send structured
        information to the environment, and the environment internally decides how to
        handle it, based on pre-registered handlers.

        This allows the user to extend the environment's behavior in a clean,
        decoupled way, without having to modify the core control flow.
        
        Example usage:
            self._context_handlers["EvaluationMetrics"] = self._handle_eval_metrics

        Args:
            context (dict): A dictionary with the following structure:
                {
                    "context_type": str,        # The name of the context class/type (e.g., "EvaluationMetrics")
                    "context_data": Any         # The actual payload, e.g., a dataclass instance
                }

        Note:
            - Handlers must be explicitly registered in the child class.
            - If no handler is found, the context is silently ignored.
            - This design promotes high cohesion and low coupling, and is open for extension
            without requiring changes to the base class.
        """
        context_type = context.get("context_type")
        handler = self._context_handlers.get(context_type, lambda context: print(f"The message {getattr(context, 'context_type', 'UNKNOWN')} has no handler."))
        handler(context)

    def baseline(self):
        """Returns a baseline prediction. Should be overridden if applicable.
        it should return:
        prediction, standart_deviation, ground_truth"""
        return np.nan, np.nan, np.nan

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
        return None, {}  # Pode ser ajustado para retornar um placeholder padrão


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

    

    @abstractmethod
    def get_bootstrap_observations(self, n_bootstrap: int, force_recompute: bool = True) -> Tuple[np.ndarray, float]:
        pass

    
