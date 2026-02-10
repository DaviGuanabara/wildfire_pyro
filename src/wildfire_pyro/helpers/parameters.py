# parameters.py

import optuna
from dataclasses import dataclass, replace
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataParameters:
    train_path: str
    validation_path: Optional[str]
    test_path: str


@dataclass(frozen=True)
class RuntimeParameters:
    GLOBAL_SEED: int
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

    @property
    def rollout_size(self) -> int:
        # if rollout_size <= 0, it will default to batch_size
        # TODO: Rethink about rollout_size. Maybe it should be optional and default to batch_size if not provided?
        # It must always be gratter than batch size

        if self.batch_size <= 0:
            raise ValueError("[MODEL PARAMETERS] batch_size must be greater than 0")

        return self.batch_size * 2


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


@dataclass
class RunParameters:
    data_parameters: DataParameters
    runtime_parameters: RuntimeParameters
    logging_parameters: LoggingParameters
    model_parameters: ModelParameters
    training_parameters: TrainingParameters
    test_parameters: TestParameters
