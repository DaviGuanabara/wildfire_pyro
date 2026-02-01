
import pandas as pd
from dataclasses import dataclass
import time
import numpy as np
import optuna
from wildfire_pyro.common.seed_manager import configure_seed_manager
from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment
from wildfire_pyro.factories.learner_factory import create_deep_set_learner, create_deep_set_learner_from_run_config

from wildfire.experiments.iowa_soil.config import RunConfig, build_run_config_from_yaml
from wildfire.experiments.iowa_soil.run import run

from pathlib import Path

DEFAULT_CONFIG_PATH = (
    Path(__file__).parent / "default_config.yaml"
).resolve()


import csv
from pathlib import Path
from typing import Dict, Any


def append_trial_result_xlsx(
    xlsx_path: Path,
    row: dict,
):
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])

    if xlsx_path.exists():
        df_old = pd.read_excel(xlsx_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(xlsx_path, index=False)



@dataclass(frozen=True)
class TrialParameters:
    hidden: int
    dropout_prob: float
    lr: float
    batch_size: int


def build_run_config_for_trial(
    *,
    trial_params: TrialParameters,
    log_dir: str,
    seed: int,

) -> RunConfig:
    """
    Build a RunConfig for a single trial by combining:
    - static defaults from YAML
    - trial-specific hyperparameters
    """

    return build_run_config_from_yaml(
        yaml_path=str(DEFAULT_CONFIG_PATH),
        log_dir=log_dir,
        seed=seed,
        lr=trial_params.lr,
        hidden=trial_params.hidden,
        dropout_prob=trial_params.dropout_prob,
        batch_size=trial_params.batch_size,
    )

def build_trial_parameters_from_trial(trial: optuna.Trial) -> TrialParameters:
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden = trial.suggest_categorical("hidden", [64, 128, 256, 512, 1024])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    return TrialParameters(
        hidden=hidden,
        dropout_prob=dropout,
        lr=lr,
        batch_size=batch_size,
    )


def _build_config(trial: optuna.Trial, log_dir: str, seed: int) -> RunConfig:
    trial_params_base = build_trial_parameters_from_trial(
        trial=trial,
    )

    trial_params = TrialParameters(
        hidden=trial_params_base.hidden,
        dropout_prob=trial_params_base.dropout_prob,
        lr=trial_params_base.lr,
        batch_size=trial_params_base.batch_size,
    )

    return build_run_config_for_trial(
        trial_params=trial_params,
        log_dir=log_dir,
        seed=seed,
    )

def gen_random_seed() -> int:
    return np.random.randint(1_000, 9_999)

def gen_environments_from_config(config: RunConfig):
    train_env = IowaEnvironment(
        data_path=config.data_parameters.train_path,
        verbose=False,
    )

    test_env = IowaEnvironment(
        data_path=config.data_parameters.test_path,
        scaler=train_env.get_fitted_scaler(),
        verbose=False,
    )
    return train_env, test_env


def objective(trial: optuna.Trial) -> float:
    start_time = time.time()

    metrics_per_run = []

    for _ in range(10):

        seed = gen_random_seed()

        # ✅ ESSENCIAL: configurar SeedManager
        configure_seed_manager(seed)

        config = _build_config(
            trial=trial,
            log_dir=f"logs/optuna/trial_{trial.number}",
            seed=seed,
        )

        train_env, eval_env = gen_environments_from_config(config)

        learner = create_deep_set_learner_from_run_config(
            train_env,
            config,
        )

        _, metrics = run(
            train_environment=train_env,
            eval_environment=eval_env,
            deep_set_learner=learner,
            config=config,
        )

        metrics_per_run.append(metrics.model_error)

    mean_error = float(np.mean(metrics_per_run))
    std_error = float(np.std(metrics_per_run))
    elapsed = time.time() - start_time

    trial.set_user_attr("elapsed_time_sec", elapsed)
    trial.set_user_attr("std_error", std_error)

    results_path = Path("logs/optuna/results.xlsx")

    append_trial_result_xlsx(
        xlsx_path=results_path,
        row={
            "trial_number": trial.number,
            "lr": trial.params["lr"],
            "hidden": trial.params["hidden"],
            "dropout": trial.params["dropout"],
            "batch_size": trial.params["batch_size"],
            "mean_error": mean_error,
            "std_error": std_error,
            "elapsed_time_sec": elapsed,
            "n_seeds": len(metrics_per_run),
        },
    )


    return mean_error
