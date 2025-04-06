import os
import torch
import numpy as np
from wildfire_pyro.environments.sensor_environment import SensorEnvironment
from wildfire_pyro.factories.learner_factory import (
    create_deep_set_learner,
)

from wildfire_pyro.common.callbacks import BootstrapEvaluationCallback


# ==================================================================================================
# Funções adicionais
# ==================================================================================================


def get_path(file_name):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join("data", "synthetic", "fixed_sensor", file_name)
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

# Setup environments
seed = 0
max_steps = 200000
n_neighbors_min = 2
n_neighbors_max = 5
verbose = False

train_data = get_path("fixed_train.csv")
validation_data = get_path("fixed_val.csv")
test_data = get_path("fixed_test.csv")


# Setup Training
total_training_steps = 20_000
n_bootstrap = 2
n_eval = 5

# Setup Evaluation
evaluations = 100

# Setup Agent Parameters
agent_parameters = {
    "lr": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dropout_prob": 0.2,
    "hidden": 64,
    "batch_size": 128,
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
    best_model_save_path="./learning_example/logs/",
    log_path="./learning_example/logs/",
    tensorboard_log="./learning_example/logs/tensorboard",
    eval_freq=1_000,
    n_eval=n_eval,
    n_bootstrap=n_bootstrap,
)


deep_set_learner = create_deep_set_learner(train_environment, agent_parameters)


# ==================================================================================================
# LEARNING
#
# TODO: Add multiple environments for training to boost performance ?
# each environment in its own thread
# ==================================================================================================

train_environment.reset(seed)
deep_set_learner.learn(
    total_timesteps=total_training_steps, callback=eval_callback, progress_bar=True
)

train_environment.close()
validation_environment.close()
print("Aprendizagem concluída")


# ==================================================================================================
# Evaluation.
# ==================================================================================================

print("\n=== Starting Bootstrap Evaluation ===")
observation, info = test_environment.reset()
wins = []
metrics = {}
metrics["sensor_id"] = info["sensor"]["sensor_id"]

for step in range(evaluations):

    bootstrap_observations, ground_truth = test_environment.get_bootstrap_observations(
        n_bootstrap
    )

    metrics["baseline_prediction"], metrics["baseline_std"], _ = (
        test_environment.baseline()
    )

    actions, _ = deep_set_learner.predict(bootstrap_observations)
    predictions = actions.squeeze().tolist()

    metrics["mean_prediction"] = np.mean(predictions)
    metrics["std_prediction"] = np.std(predictions)
    metrics["error"] = metrics["mean_prediction"] - ground_truth
    metrics["baseline_error"] = metrics["baseline_prediction"] - ground_truth
    metrics["step"] = step

    wins.append(metrics["error"] < metrics["baseline_error"])

    log_evaluation(metrics, info, step)

    observation, reward, terminated, truncated, info = test_environment.step(
        np.array(metrics["mean_prediction"])
    )

    if terminated:
        print("\nThe episode has ended.")
        break

print(f"Final of evaluation. Win Rate: {np.mean(wins)}")

test_environment.close()
