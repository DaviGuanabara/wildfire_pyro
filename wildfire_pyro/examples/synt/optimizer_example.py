import os
import torch
import numpy as np
from wildfire_pyro.environments.sensor.sensor_environment import SensorEnvironment
from wildfire_pyro.factories.learner_factory import (
    create_deep_set_learner,
    SupervisedLearningManager,
)

from wildfire_pyro.common.callbacks import BootstrapEvaluationCallback


print("Em construção")


def get_path(file_name):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("data", "synthetic", "fixed_sensor", file_name)
    data_path = os.path.join(SCRIPT_DIR, relative_path)
    return data_path


# ==================================================================================================
# SETUP
# Configurações do ambiente de treinamento e teste
# Configurações do agente DeepSet
# ==================================================================================================


def training():
    agent_parameters = {
        "lr": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dropout_prob": 0.2,
        "hidden": 64,
        "batch_size": 128,
    }

    seed = 0
    max_steps = 200000
    n_neighbors_min = 2
    n_neighbors_max = 5

    total_training_steps = 20_000
    n_bootstrap = 2
    n_eval = 5

    train_data = get_path("fixed_train.csv")
    validation_data = get_path("fixed_val.csv")
    test_data = get_path("fixed_test.csv")

    train_environment = SensorEnvironment(
        data_path=train_data,
        max_steps=max_steps,
        n_neighbors_min=n_neighbors_min,
        n_neighbors_max=n_neighbors_max,
    )

    validation_environment = SensorEnvironment(
        data_path=validation_data,
        max_steps=max_steps,
        n_neighbors_min=n_neighbors_min,
        n_neighbors_max=n_neighbors_max,
    )

    test_environment = SensorEnvironment(
        data_path=test_data,
        max_steps=max_steps,
        n_neighbors_min=n_neighbors_min,
        n_neighbors_max=n_neighbors_max,
    )

    eval_callback = BootstrapEvaluationCallback(
        validation_environment,
        best_model_save_path="./logs/",
        eval_freq=1_000,
        n_eval=n_eval,
        n_bootstrap=n_bootstrap,
    )

    deep_set = create_deep_set_learner(
        train_environment,
        model_parameters=agent_parameters,
        logging_parameters={"tensorboard_log": "./logs/tensorboard"},
        runtime_parameters={
            "device": agent_parameters["device"], "verbose": 1},
    )


    # ==================================================================================================
    # LEARNING
    # Executa o processo de aprendizagem
    # TODO: Add multiple environments for training to boost performance ?
    # each environment in its own thread
    # ==================================================================================================

    train_environment.reset(seed)
    deep_set.learn(
        total_timesteps=total_training_steps, callback=eval_callback, progress_bar=True
    )

    train_environment.close()
    validation_environment.close()
    print("Aprendizagem concluída")

    # ==================================================================================================
    # INFERENCE
    # Teste de inferência após o treinamento com Bootstrap
    # O treinamento segue a ideia de gerar N conjuntos de vizinhos para estimar a incerteza.
    # ==================================================================================================

    print("\n=== Starting Bootstrap Evaluation ===")
    observation, info = test_environment.reset()
    wins = []
    evaluations = 100

    for step in range(evaluations):

        bootstrap_observations, ground_truth = (
            test_environment.get_bootstrap_observations(n_bootstrap)
        )

        baseline_prediction, baseline_std, baseline_ground_truth = (
            test_environment.baseline()
        )

        actions, _ = deep_set.predict(bootstrap_observations)
        predictions = actions.squeeze().tolist()

        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        error = mean_prediction - ground_truth
        baseline_error = baseline_prediction - ground_truth

        wins.append(error < baseline_error)

        # Move to the next sensor
        final_prediction = np.array([mean_prediction])
        observation, reward, terminated, truncated, info = test_environment.step(
            final_prediction
        )

        if terminated:
            break

    test_environment.close()
    return np.mean(wins)
