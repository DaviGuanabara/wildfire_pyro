import torch
import numpy as np
from typing import Any, Dict, Tuple

from gymnasium import spaces

from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.supervised_learning_manager import (
    SupervisedLearningManager,
    BaseLearningManager,
)


def create_deep_set_learner(
    env: BaseEnvironment,
    model_parameters: Dict[str, Any],
    logging_parameters: Dict[str, Any],
    runtime_parameters: Dict[str, Any],
) -> SupervisedLearningManager:
    """
    Factory function to instantiate the DeepSetAttentionNet model and its LearningManager.

    Args:
        env: The environment used for observation and action space definitions.
        model_parameters: Parameters specific to the neural model (e.g., lr, hidden, dropout).
        logging_parameters: Logger configuration (e.g., log_path, tensorboard_log).
        runtime_parameters: Device, seed, verbosity and other runtime controls.

    Returns:
        SupervisedLearningManager instance, fully configured.
    """

    # Determine input and output dimensions from environment
    if isinstance(env.observation_space, spaces.Box):
        input_dim = env.observation_space.shape[-1]
    elif isinstance(env.observation_space, spaces.Discrete):
        input_dim = 1
    else:
        raise NotImplementedError(
            f"Unsupported observation space: {type(env.observation_space)}"
        )

    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[-1]
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
    else:
        raise NotImplementedError(f"Unsupported action space: {type(env.action_space)}")

    # Cria modelo com base em model_parameters
    neural_network = DeepSetAttentionNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden=model_parameters.get("hidden", 64),
        prob=model_parameters.get("dropout_prob", 0.2),
    ).to(runtime_parameters.get("device", "cpu"))

    # Instancia o learner
    learner = SupervisedLearningManager(
        neural_network=neural_network,
        environment=env,
        logging_parameters=logging_parameters,
        runtime_parameters=runtime_parameters,
        model_parameters=model_parameters,
    )

    return learner
