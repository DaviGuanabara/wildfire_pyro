import time
import optuna
from pathlib import Path

from wildfire.experiments.iowa_soil.objective import objective


def eta_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and "elapsed_time_sec" in t.user_attrs
    ]

    if len(completed) < 2:
        return

    times = [t.user_attrs["elapsed_time_sec"] for t in completed]
    avg_time = sum(times) / len(times)

    total_trials = study.user_attrs.get("n_trials_target")
    if total_trials is None:
        return

    remaining = total_trials - len(completed)
    eta_sec = remaining * avg_time

    print(
        f"[ETA] avg/trial: {avg_time:.1f}s | "
        f"remaining: {remaining} | "
        f"ETA: {eta_sec/60:.1f} min ({eta_sec/3600:.2f} h)"
    )


def run_optuna(
    *,
    n_trials: int = 30,
    study_name: str = "iowa_soil_optuna",
    storage_path: str = "logs/optuna/study.db",
    n_jobs: int = 1,
):

    storage_path_obj = Path(storage_path)
    storage_path_obj.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{storage_path_obj}",
        load_if_exists=True,
    )

    study.set_user_attr("n_trials_target", n_trials)

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        callbacks=[eta_callback],
        show_progress_bar=True,
    )

    print("\n=== OPTUNA FINISHED ===")
    print(f"Best value (mean error): {study.best_value:.6f}")
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print("\nExtra stats:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")

    return study


if __name__ == "__main__":
    run_optuna()
