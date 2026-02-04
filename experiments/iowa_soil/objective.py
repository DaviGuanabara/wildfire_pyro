import time
import optuna
import pandas as pd
from pathlib import Path

from iowa_experiment import IowaEnvironmentExperiment
from runtime_config import BASE_RUN_PARAMETERS


def objective(trial: optuna.Trial) -> float:
    start = time.time()

    run_parameters = BASE_RUN_PARAMETERS.with_trial(trial)

    experiment = IowaEnvironmentExperiment(run_parameters)
    _, metrics = experiment.run()

    elapsed = time.time() - start
    trial.set_user_attr("elapsed_time_sec", elapsed)

    path = Path("logs/optuna/results.xlsx")
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
