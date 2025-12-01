from abc import ABC, abstractmethod


class BaselineStrategy(ABC):
    """
    Abstract base class for all baseline strategies.
    """

    def __init__(self, observation_space, action_space, scaler=None):
        self.observation_space = observation_space
        self.action_space = action_space
        self.scaler = scaler

    @abstractmethod
    def predict(self, observations):
        """Return baseline predictions."""
        pass

    @abstractmethod
    def set_schema(self, neighbor_schema):
        """Set the schema for neighbor data."""
        pass
