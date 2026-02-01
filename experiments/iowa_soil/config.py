from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import torch
import yaml


# =========================
# Dataclasses
# =========================

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


# =========================
# Builder
# =========================

def build_run_config_from_yaml(
    yaml_path: str,
    *,
    seed: int,
    log_dir: str,
    lr: float,
    hidden: int,
    batch_size: int,
    dropout_prob: float,
) -> RunConfig:

    yaml_path_obj = Path(yaml_path).resolve()
    base_dir = yaml_path_obj.parent

    if not yaml_path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path_obj}")

    with open(yaml_path_obj, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    # ------------------------
    # Runtime
    # ------------------------
    runtime_parameters = RuntimeParameters(
        seed=seed,
        log_dir=log_dir,
        verbose=spec["runtime"]["verbose"],
        device=spec["runtime"].get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ),
    )

    # ------------------------
    # Logging
    # ------------------------
    logging_parameters = LoggingParameters(
        log_path=log_dir,
        format_strings=tuple(spec["logging"]["format_strings"]),
    )

    # ------------------------
    # Model
    # ------------------------
    model_parameters = ModelParameters(
        lr=lr,
        hidden=hidden,
        batch_size=batch_size,
        dropout_prob=dropout_prob,
    )

    # ------------------------
    # Training
    # ------------------------
    training = spec["training"]
    training_parameters = TrainingParameters(
        total_timesteps=training["total_timesteps"],
        use_validation=training["use_validation"],
        eval_freq=training["eval_freq"],
        log_frequency=training["log_frequency"],
    )

    # ------------------------
    # Evaluation
    # ------------------------
    evaluation = spec["evaluation"]
    test_parameters = TestParameters(
        n_eval=evaluation["n_eval"],
        n_bootstrap=evaluation["n_bootstrap"],
    )

    # ------------------------
    # Data (paths resolvidos)
    # ------------------------
    data = spec["data"]
    data_parameters = DataParameters(
        train_path=str((base_dir / data["train_path"]).resolve()),
        validation_path=(
            str((base_dir / data["validation_path"]).resolve())
            if data.get("validation_path") is not None
            else None
        ),
        test_path=str((base_dir / data["test_path"]).resolve()),
    )

    return RunConfig(
        runtime_parameters=runtime_parameters,
        logging_parameters=logging_parameters,
        model_parameters=model_parameters,
        training_parameters=training_parameters,
        test_parameters=test_parameters,
        data_parameters=data_parameters,
    )
