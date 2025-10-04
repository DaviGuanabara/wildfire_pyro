import os
import torch
import numpy as np
from datetime import datetime

from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment
from wildfire_pyro.factories.learner_factory import create_deep_set_learner
from wildfire_pyro.common.callbacks import BootstrapEvaluationCallback
from wildfire_pyro.common.seed_manager import configure_seed_manager, get_seed
from wildfire_pyro.helpers.custom_schedulers import DebugScheduler
from wildfire_pyro.helpers.learning_utils import log_evaluation


# ==================================================================================================
# SETUP
# ==================================================================================================

global_seed = 42
configure_seed_manager(global_seed)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_data = os.path.join(BASE_DIR, "data", "train.csv")
validation_data = os.path.join(BASE_DIR, "data", "val.csv")
test_data = os.path.join(BASE_DIR, "data", "test.csv")

# Setup Training
total_training_steps = 200_000
n_bootstrap = 2
n_eval = 5
evaluations = 100
verbose = False

model_parameters = {
    "lr": DebugScheduler(lr=0.001, verbose=verbose),
    "dropout_prob": 0.2,
    "hidden": 64,
    "batch_size": 256,
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
# INSTANTIATION
# ==================================================================================================

train_environment = IowaEnvironment(data_path=train_data, verbose=verbose)
validation_environment = IowaEnvironment(data_path=validation_data, verbose=verbose)
test_environment = IowaEnvironment(data_path=test_data, verbose=verbose)

eval_callback = BootstrapEvaluationCallback(
    validation_environment,
    best_model_save_path=logging_parameters.get("log_path"),
    seed=get_seed("Bootstrap_Evaluation_Callback"),
    eval_freq=10_000,
    n_eval=n_eval,
    n_bootstrap=n_bootstrap,
    verbose=verbose,
)

deep_set_learner = create_deep_set_learner(
    train_environment, model_parameters, logging_parameters, runtime_parameters
)


# ==================================================================================================
# LEARNING
# ==================================================================================================

train_environment.reset(runtime_parameters.get("seed", 42))
deep_set_learner.learn(
    total_timesteps=total_training_steps, callback=eval_callback, progress_bar=True
)

train_environment.close()
validation_environment.close()
print("Aprendizagem concluída")


# ==================================================================================================
# EVALUATION
# ==================================================================================================

print("\n=== Starting Bootstrap Evaluation ===")
print("In maintenance...")
observation, info = test_environment.reset(seed=get_seed("test"))
wins = []
metrics = {}

# Como o IowaEnvironment não tem "sensor_id", você pode adaptar:
metrics["dataset_id"] = info["sample"][test_environment.dataset_metadata.id]

for step in range(evaluations):

    bootstrap_observations, ground_truth = test_environment.get_bootstrap_observations(
        n_bootstrap
    )

test_environment.close()
