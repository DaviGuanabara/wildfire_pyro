from dataclasses import asdict, dataclass
import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple

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

    # Define output_dim a partir do action_space
    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[-1]
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
    else:
        raise NotImplementedError(
            f"Unsupported action space: {type(env.action_space)}")

    # Cria modelo (rede neural decide input_dim internamente a partir do Dict)
    neural_network = DeepSetAttentionNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=model_parameters.get("hidden", 64),
        prob=model_parameters.get("dropout_prob", 0.2),

    ).to(runtime_parameters.get("device", "cpu"))

    learner = SupervisedLearningManager(
        neural_network=neural_network,
        environment=env,
        logging_parameters=logging_parameters,
        runtime_parameters=runtime_parameters,
        model_parameters=model_parameters,
    )

    return learner



