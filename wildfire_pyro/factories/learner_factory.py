import torch
import numpy as np
from typing import Any, Dict, Tuple

from gymnasium import spaces

from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.supervised_learning_manager import SupervisedLearningManager, BaseLearningManager


def create_deep_set_learner(env: BaseEnvironment, parameters: Dict[str, Any]) -> SupervisedLearningManager:
    """
    Factory function to instantiate the DeepSetAttentionNet model and its LearningManager.

    Args:
        env (Any): Gymnasium environment instance.
        parameters (Dict[str, Any]): Dictionary containing model and training parameters.

    Returns:
        LearningManager: Configured model manager for training and inference.
    """
    # Determine input dimensions from the environment's observation space
    if isinstance(env.observation_space, spaces.Box):
        input_dim = env.observation_space.shape[-1]
    elif isinstance(env.observation_space, spaces.Discrete):
        input_dim = 1
    else:
        raise NotImplementedError(
            f"Unsupported observation space: {type(env.observation_space)}"
        )

    # Determine output dimensions from the environment's action space
    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[-1]
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
    else:
        raise NotImplementedError(
            f"Unsupported action space: {type(env.action_space)}")

    # Configure additional parameters
    parameters["input_dim"] = input_dim
    parameters["output_dim"] = output_dim

    # Instantiate the neural network based on environment dimensions
    neural_network = DeepSetAttentionNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden=parameters.get("hidden", 32),
        prob=parameters.get("dropout_prob", 0.5),
    ).to(parameters.get("device", "cpu"))

    # Instantiate the LearningManager with buffer size equal to batch_size
    learning_manager = SupervisedLearningManager(
        neural_network=neural_network,
        environment=env,
        parameters=parameters,
        batch_size=parameters.get("batch_size", 64),
    )

    return learning_manager