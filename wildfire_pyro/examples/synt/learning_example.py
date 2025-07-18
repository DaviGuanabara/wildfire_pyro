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
from wildfire_pyro.helpers.learning_utils import get_path, log_evaluation



# ==================================================================================================
# SETUP
# Configurações do ambiente de treinamento e teste
# Configurações do agente DeepSet
#
# TODO: environment_parameters dictionary
# TODO: training_parameters dictionary
# ==================================================================================================


# just need to set it once, for reproducibility
global_seed = 42
configure_seed_manager(global_seed)

train_data = get_path("fixed_train.csv")
validation_data = get_path("fixed_val.csv")
test_data = get_path("fixed_test.csv")

# Setup environments
max_steps = 200000
n_neighbors_min = 5 # 20 no taxi bj
n_neighbors_max = 30 # 50 no taxi bj
verbose = False

# Setup Training
total_training_steps = 200_000
n_bootstrap = 2
n_eval = 5 # 1600 ?

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
# TODO: Early stopping.
# TODO: Maybe load the best model generated ?
# ==================================================================================================

train_environment.reset(runtime_parameters.get("seed", 42))
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
observation, info = test_environment.reset(seed=get_seed("test"))
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
        observation, info = test_environment.reset(seed=get_seed("test"))
        break

print(f"Final of evaluation. Win Rate: {np.mean(wins)}")

test_environment.close()
