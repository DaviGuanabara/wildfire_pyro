import os
import torch
import numpy as np
from wildfire_pyro.environments.sensor.sensor_environment import SensorEnvironment
from wildfire_pyro.factories.learner_factory import (
    create_deep_set_learner,
)

from wildfire_pyro.common.callbacks import BootstrapEvaluationCallback
from datetime import datetime
from wildfire_pyro.common.seed_manager import configure_seed_manager, get_seed

from wildfire_pyro.helpers.custom_schedulers import DebugScheduler


"""
TODO: Early stopping.
"""

import cProfile


def main():



    # ==================================================================================================
    # Funções adicionais
    # ==================================================================================================


    def get_path(file_name):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(
            "data", "synthetic", "fixed_sensor", file_name)
        data_path = os.path.join(SCRIPT_DIR, relative_path)
        return data_path


    def log_evaluation(metrics, info, step):
        print(f"\n--- Step {step} ---")
        print(f">> Evaluating Sensor (ID: {info["sensor"]['sensor_id']})")
        print(
            f"   Location: Latitude {info["sensor"]['lat']:.4f}, Longitude: {info["sensor"]['lon']:.4f}, Ground Truth: {info['ground_truth']:.4f}"
        )
        print(">> Bootstrap Model:")
        print(
            f"   Prediction: {metrics['mean_prediction']:.4f} ± {metrics['std_prediction']:.4f} | Error: {metrics['error']:.4f}"
        )
        print(">> Baseline:")
        print(
            f"   Prediction: {metrics['baseline_prediction']:.4f} ± {metrics['baseline_std']:.4f} | Error: {metrics['baseline_error']:.4f}"
        )


    # ==================================================================================================
    # SETUP
    # Configurações do ambiente de treinamento e teste
    # Configurações do agente DeepSet
    # ==================================================================================================


    # just need to set it once, for reproducibility
    global_seed = 42
    configure_seed_manager(global_seed)

    train_data = get_path("fixed_train.csv")
    validation_data = get_path("fixed_val.csv")
    test_data = get_path("fixed_test.csv")

    # Setup environments
    max_steps = 200000
    n_neighbors_min = 5  # 20 no taxi bj
    n_neighbors_max = 30  # 50 no taxi bj
    verbose = False

    # Setup Training
    total_training_steps = 10_000
    n_bootstrap = 2
    n_eval = 5  # 1600 ?

    # Setup Evaluation
    evaluations = 100


    model_parameters = {
        "lr": DebugScheduler(lr=0.001, verbose=verbose),
        "dropout_prob": 0.2,
        "hidden": 64,
        "batch_size": 128,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    log_dir = os.path.join("./logs", run_id)

    logging_parameters = {
        "log_path": log_dir,
        "format_strings": ["csv", "tensorboard", "stdout"],
    }

    runtime_parameters = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": get_seed(f"run_{0}"),
        "verbose": verbose,
    }


    # ==================================================================================================
    # Instantiation
    # ==================================================================================================


    train_environment = SensorEnvironment(
        data_path=train_data,
        max_steps=max_steps,
        n_neighbors_min=n_neighbors_min,
        n_neighbors_max=n_neighbors_max,
        verbose=verbose,
    )

    validation_environment = SensorEnvironment(
        data_path=validation_data,
        max_steps=max_steps,
        n_neighbors_min=n_neighbors_min,
        n_neighbors_max=n_neighbors_max,
        verbose=verbose,
    )

    test_environment = SensorEnvironment(
        data_path=test_data,
        max_steps=max_steps,
        n_neighbors_min=n_neighbors_min,
        n_neighbors_max=n_neighbors_max,
        verbose=verbose,
    )

    eval_callback = BootstrapEvaluationCallback(
        validation_environment,
        best_model_save_path=logging_parameters.get("log_path"),
        seed=get_seed("Bootstrap_Evaluation_Callback"),
        eval_freq=5_000,
        n_eval=n_eval,
        n_bootstrap=n_bootstrap,
        verbose=verbose,
    )


    deep_set_learner = create_deep_set_learner(
        train_environment, model_parameters, logging_parameters, runtime_parameters
    )


    # ==================================================================================================
    # LEARNING
    #
    # TODO: Add multiple environments for training to boost performance ?
    # each environment in its own thread
    # ==================================================================================================

    train_environment.reset(runtime_parameters.get("seed", 42))
    deep_set_learner.learn(
        total_timesteps=total_training_steps, callback=eval_callback, progress_bar=True
    )

    train_environment.close()
    validation_environment.close()
    print("Aprendizagem concluída")




cProfile.run("main()", sort="cumtime", filename="profile.prof")
