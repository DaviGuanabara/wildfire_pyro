from copy import deepcopy
import time
import optuna
import pandas as pd
from pathlib import Path

from wildfire_pyro.helpers.parameters import ModelParameters, RunParameters

from iowa_experiment import IowaEnvironmentExperiment
from runtime_config import BASE_RUN_PARAMETERS


def _gen_run_parameters(trial: optuna.Trial) -> RunParameters:
    run_parameters = deepcopy(BASE_RUN_PARAMETERS)

    lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    hidden = trial.suggest_categorical("hidden", [64, 128, 256, 512])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])

    run_parameters.logging_parameters.log_folder = f"trial_{trial.number}"

    run_parameters.model_parameters = ModelParameters(
        lr=lr, hidden=hidden, dropout_prob=dropout, batch_size=batch_size
    )

    return run_parameters


def objective(trial: optuna.Trial) -> float:

    start = time.time()

    run_parameters = _gen_run_parameters(trial)

    experiment = IowaEnvironmentExperiment(run_parameters)
    _, metrics = experiment.run()

    elapsed = time.time() - start
    trial.set_user_attr("elapsed_time_sec", elapsed)

    log_dir = Path(BASE_RUN_PARAMETERS.logging_parameters.log_dir)
    path = log_dir / "optuna/results.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "trial_number": trial.number,
        **trial.params,
        # Bootstrap-aware decision metrics
        "model_mae_mean": metrics.model_mae_mean,
        "model_mae_std": metrics.model_mae_std,
        "baseline_mae_mean": metrics.baseline_mae_mean,
        "baseline_mae_std": metrics.baseline_mae_std,
        "win_rate_over_baseline": metrics.win_rate_over_baseline,
        # Diagnostic
        "model_rmse_mean": metrics.model_rmse_mean,
        "baseline_rmse_mean": metrics.baseline_rmse_mean,
        "elapsed_time_sec": elapsed,
    }

    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_excel(path), df], ignore_index=True)
    df.to_excel(path, index=False)

    return metrics.model_mae_mean
