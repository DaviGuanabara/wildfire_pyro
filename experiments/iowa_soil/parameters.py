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
    base_seed: int
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
class RunParameters:
    data_parameters: DataParameters
    runtime_parameters: RuntimeParameters
    logging_parameters: LoggingParameters
    model_parameters: ModelParameters
    training_parameters: TrainingParameters
    test_parameters: TestParameters

    def with_trial(self, trial: optuna.Trial) -> "RunParameters":
        log_dir = f"logs/optuna/trial_{trial.number}"

        return replace(
            self,
            runtime_parameters=replace(
                self.runtime_parameters,
                seed=trial.number,
                log_dir=log_dir,
            ),
            logging_parameters=replace(
                self.logging_parameters,
                log_path=log_dir,
            ),
            model_parameters=replace(
                self.model_parameters,
                lr=trial.suggest_float("lr", 1e-4, 1e-1, log=True),
                hidden=trial.suggest_categorical(
                    "hidden", [64, 128, 256, 512, 1024]
                ),
                dropout_prob=trial.suggest_float("dropout", 0.0, 0.5),
                batch_size=trial.suggest_categorical(
                    "batch_size", [64, 128, 256]
                ),
            ),
        )
