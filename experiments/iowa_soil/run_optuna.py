import optuna
from pathlib import Path

from wildfire_pyro.common.seed_manager import configure_seed_manager
from optuna.samplers import TPESampler

from objective import objective
from runtime_config import OPTUNA_CONFIG, BASE_RUN_PARAMETERS


def run_optuna():

    log_dir = Path(BASE_RUN_PARAMETERS.logging_parameters.log_dir)
    storage_path = log_dir / "optuna/study.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    seed_manager = configure_seed_manager(
        BASE_RUN_PARAMETERS.runtime_parameters.GLOBAL_SEED
    )
    optuna_seed = seed_manager.get_seed("optuna_sampler")
    sampler = TPESampler(seed=optuna_seed)

    study = optuna.create_study(
        study_name=OPTUNA_CONFIG.study_name,
        direction=OPTUNA_CONFIG.direction,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=sampler,
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
