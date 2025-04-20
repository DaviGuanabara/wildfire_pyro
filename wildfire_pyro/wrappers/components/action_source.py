
class BaseActionSource:
    """
    Base class for action sources.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state):
        """
        Get an action from the action source.
        """
        raise NotImplementedError("get_action not implemented")
    
    
