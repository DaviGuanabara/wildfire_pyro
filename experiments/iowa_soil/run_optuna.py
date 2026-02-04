import optuna
from pathlib import Path

from objective import objective
from runtime_config import OPTUNA_CONFIG


def run_optuna():
    storage_path = Path("logs/optuna/study.db")
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=OPTUNA_CONFIG.study_name,
        direction=OPTUNA_CONFIG.direction,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=OPTUNA_CONFIG.n_trials,
        n_jobs=OPTUNA_CONFIG.n_jobs,
        show_progress_bar=True,
    )

    print("\n=== OPTUNA FINISHED ===")
    print(f"Best value: {study.best_value:.6f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study


if __name__ == "__main__":
    run_optuna()
