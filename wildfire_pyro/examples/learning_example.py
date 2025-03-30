import os
import torch
import numpy as np
from wildfire_pyro.environments.sensor_environment import SensorEnvironment
from wildfire_pyro.factories.learner_factory import (
    create_deep_set_learner,
)

from wildfire_pyro.common.callbacks import EvalCallback

print("Learning Example está em construção")


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


seed = 0
max_steps = 200000
n_neighbors_min = 2
n_neighbors_max = 5

train_data = get_path("fixed_train.csv")
validation_data = get_path("fixed_val.csv")
test_data = get_path("fixed_test.csv")

total_training_steps = 1_000_000
n_bootstrap = 2
n_eval = 5

agent_parameters = {
    "lr": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dropout_prob": 0.2,
    "hidden": 64,
    "batch_size": 128,
}


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


eval_callback = EvalCallback(
    validation_environment,
    best_model_save_path="./logs/",
    log_path="./logs/",
    tensorboard_log="./logs/tensorboard",
    eval_freq=10_000,
    n_eval=n_eval,
    n_bootstrap=n_bootstrap,
)


deep_set = create_deep_set_learner(train_environment, agent_parameters)


# ==================================================================================================
# LEARNING
# Executa o processo de aprendizagem
# TODO: Add multiple environments for training to boost performance
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

for step in range(2):

    bootstrap_observations, ground_truth = test_environment.get_bootstrap_observations(
        n_bootstrap
    )

    predictions = []
    for obs in bootstrap_observations:
        action, _ = deep_set.predict(obs)
        predictions.append(action.item())

    mean_prediction = np.mean(predictions)
    std_prediction = np.std(predictions)
    error = mean_prediction - ground_truth

    print(f"\n--- Step {step} ---")
    print(f">> Evaluating Sensor (ID: {info['sensor']['sensor_id']})")
    print(
        f"   Location: Latitude {info['sensor']['lat']:.4f}, Longitude {info['sensor']['lon']:.4f}, y {info['ground_truth']:.4f}"
    )
    print(
        f">> Bootstrap Results: Mean Prediction: {mean_prediction:.4f}, "
        f"Std Dev: {std_prediction:.4f}, Ground Truth: {ground_truth:.4f}, "
        f"Error: {error:.4f}"
    )

    # Move to the next sensor
    final_prediction = np.array([mean_prediction])
    observation, reward, terminated, truncated, info = test_environment.step(
        final_prediction
    )

    if terminated:
        print("\nThe episode has ended.")
        break


test_environment.close()
