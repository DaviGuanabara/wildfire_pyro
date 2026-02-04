# config.py
from dataclasses import dataclass
from pathlib import Path
import torch
import yaml

from parameters import (
    RunParameters,
    RuntimeParameters,
    LoggingParameters,
    ModelParameters,
    TrainingParameters,
    TestParameters,
    DataParameters,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "default_config.yaml"
).resolve()


@dataclass(frozen=True)
class OptunaConfig:
    n_trials: int
    study_name: str
    n_jobs: int
    direction: str = "minimize"


@dataclass(frozen=True)
class LoadedConfig:
    base_run_parameters: RunParameters
    optuna: OptunaConfig


def load_full_config(yaml_path: str) -> LoadedConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    base_dir = Path(yaml_path).parent

    runtime = spec["runtime"]
    runtime_parameters = RuntimeParameters(
        base_seed=runtime["base_seed"],
        seed=runtime["base_seed"],   # inicial, será derivada depois
        log_dir="",
        verbose=runtime["verbose"],
        device=runtime.get("device", "cpu")
    )


    logging_parameters = LoggingParameters(
        log_path="",
        format_strings=tuple(spec["logging"]["format_strings"]),
    )

    model_parameters = ModelParameters(
        lr=0.0,
        hidden=0,
        batch_size=0,
        dropout_prob=0.0,
    )

    training = spec["training"]
    training_parameters = TrainingParameters(
        total_timesteps=training["total_timesteps"],
        use_validation=training["use_validation"],
        log_frequency=training["log_frequency"],
        eval_freq=training.get("eval_freq"),
    )

    evaluation = spec["evaluation"]
    test_parameters = TestParameters(
        n_eval=evaluation["n_eval"],
        n_bootstrap=evaluation["n_bootstrap"],
    )

    data = spec["data"]
    data_parameters = DataParameters(
        train_path=str((base_dir / data["train_path"]).resolve()),
        validation_path=(
            str((base_dir / data["validation_path"]).resolve())
            if data.get("validation_path") else None
        ),
        test_path=str((base_dir / data["test_path"]).resolve()),
    )

    base_run_parameters = RunParameters(
        data_parameters=data_parameters,
        runtime_parameters=runtime_parameters,
        logging_parameters=logging_parameters,
        model_parameters=model_parameters,
        training_parameters=training_parameters,
        test_parameters=test_parameters,
    )

    optuna_spec = spec["optuna"]
    optuna_config = OptunaConfig(
        n_trials=optuna_spec["n_trials"],
        study_name=optuna_spec["study_name"],
        n_jobs=optuna_spec.get("n_jobs", 1),
        direction=optuna_spec.get("direction", "minimize"),
    )

    return LoadedConfig(
        base_run_parameters=base_run_parameters,
        optuna=optuna_config,
    )
