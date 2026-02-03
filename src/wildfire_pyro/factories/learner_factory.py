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


@dataclass(frozen=True)
class DataParameters:
    train_path: str
    validation_path: Optional[str]
    test_path: str


@dataclass(frozen=True)
class RuntimeParameters:
    seed: int
    log_dir: str
    verbose: bool
    device: str


@dataclass(frozen=True)
class LoggingParameters:
    log_path: str
    format_strings: tuple[str, ...]


@dataclass(frozen=True)
class ModelParameters:
    lr: float
    dropout_prob: float
    hidden: int
    batch_size: int


@dataclass(frozen=True)
class TrainingParameters:
    total_timesteps: int
    use_validation: bool
    log_frequency: int
    eval_freq: Optional[int] = None


@dataclass(frozen=True)
class TestParameters:
    n_bootstrap: int
    n_eval: int


@dataclass(frozen=True)
class RunConfig:
    data_parameters: DataParameters
    runtime_parameters: RuntimeParameters
    logging_parameters: LoggingParameters
    model_parameters: ModelParameters
    training_parameters: TrainingParameters
    test_parameters: TestParameters



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


def create_deep_set_learner_from_run_config(
    env: BaseEnvironment,
    config: RunConfig
) -> SupervisedLearningManager:


    neural_network = DeepSetAttentionNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hidden_dim=config.model_parameters.hidden,
        prob=config.model_parameters.dropout_prob,

    ).to(config.runtime_parameters.device)

    learner = SupervisedLearningManager(
        neural_network=neural_network,
        environment=env,
        logging_parameters=asdict(config.logging_parameters),
        runtime_parameters=asdict(config.runtime_parameters),
        model_parameters=asdict(config.model_parameters),
    )

    return learner
