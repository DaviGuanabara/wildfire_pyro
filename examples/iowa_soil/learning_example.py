from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.common.baselines.MeanNeighborBaseline import MeanNeighborBaseline
import os
import torch
import numpy as np
from datetime import datetime

from wildfire_pyro.environments.iowa.iowa_environment import IowaEnvironment
from wildfire_pyro.factories.learner_factory import create_deep_set_learner
from wildfire_pyro.common.callbacks import BootstrapEvaluationCallback, CallbackList, TrainLoggingCallback
from wildfire_pyro.common.seed_manager import configure_seed_manager, get_seed
from wildfire_pyro.helpers.custom_schedulers import DebugScheduler
from wildfire_pyro.helpers.learning_utils import log_evaluation


# ==================================================================================================
# SETUP
# ==================================================================================================

global_seed = 42
configure_seed_manager(global_seed)

data_path_windows = "C:\\Users\\davi_\\Documents\\GitHub\\wildfire_workspace\\wildfire\\examples\\iowa_soil\\data\\daily\\processed\\dataset_with_baseline.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



train_data = os.path.join(BASE_DIR, "data", "train.csv")
validation_data = os.path.join(BASE_DIR, "data", "val.csv")
test_data = os.path.join(BASE_DIR, "data", "test.csv")

# Setup Training
total_training_steps = 200_000
n_bootstrap = 2
n_eval = 5
#evaluations = 100
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
    "seed": get_seed("run_0"),
    "verbose": verbose,
}


# ==================================================================================================
# INSTANTIATION
# “Normalization parameters were estimated exclusively from the training set and 
# reused unchanged for validation and test environments via explicit scaler injection.”
# ==================================================================================================

train_environment = IowaEnvironment(data_path=train_data, verbose=verbose, baseline_type="mean_neighbor")
standard_scaler = train_environment.get_fitted_scaler()

validation_environment = IowaEnvironment(data_path=validation_data, verbose=verbose, baseline_type="mean_neighbor", scaler=standard_scaler)
test_environment = IowaEnvironment(data_path=test_data, verbose=verbose, baseline_type="mean_neighbor", scaler=standard_scaler)

eval_callback = BootstrapEvaluationCallback(
    validation_environment,
    best_model_save_path=logging_parameters.get("log_path"),
    seed=get_seed("Bootstrap_Evaluation_Callback"),
    eval_freq=10_000,
    n_eval=n_eval,
    n_bootstrap=n_bootstrap,
    verbose=verbose,
)

train_callback = TrainLoggingCallback(log_freq=1000, verbose=True)

callbacks = CallbackList([train_callback, eval_callback])

deep_set_learner = create_deep_set_learner(
    train_environment, model_parameters, logging_parameters, runtime_parameters
)


# ==================================================================================================
# LEARNING
# ==================================================================================================

train_environment.reset(runtime_parameters.get("seed", 42))
deep_set_learner.learn(
    total_timesteps=total_training_steps, callback=callbacks, progress_bar=True
)

train_environment.close()
validation_environment.close()
print("Aprendizagem concluída")


# ==================================================================================================
# EVALUATION
# ==================================================================================================


print("\n=== Starting Bootstrap Evaluation ===")
print("In maintenance...")

print("\n=== FINAL TEST EVALUATION (BOOTSTRAP) ===")

test_environment.reset(seed=get_seed("test_final"))

nn_mean_errors = []
baseline_mean_errors = []
comparisons = []

for _ in range(n_eval):
    bootstrap_obs, ground_truths, baselines = (
        test_environment.get_bootstrap_observations(n_bootstrap)
    )

    preds, _ = deep_set_learner.predict(bootstrap_obs)

    pred_tensor = torch.tensor(preds, dtype=torch.float32)
    gt_tensor = torch.tensor(ground_truths, dtype=torch.float32)
    base_tensor = torch.tensor(baselines, dtype=torch.float32)

    loss_fn = torch.nn.MSELoss()

    nn_error = loss_fn(pred_tensor, gt_tensor).item()
    baseline_error = loss_fn(base_tensor, gt_tensor).item()

    nn_mean_errors.append(nn_error)
    baseline_mean_errors.append(baseline_error)
    comparisons.append(int(nn_error < baseline_error))

mean_model_error = float(np.mean(nn_mean_errors))
std_model_error = float(np.std(nn_mean_errors))
mean_baseline_error = float(np.mean(baseline_mean_errors))
std_baseline_error = float(np.std(baseline_mean_errors))
win_rate = float(np.mean(comparisons))

print("\n=== FINAL RESULTS ===")
print(
    f"Model MAE (bootstrap):     {mean_model_error:.4f} ± {std_model_error:.4f}")
print(
    f"Baseline MAE (bootstrap):  {mean_baseline_error:.4f} ± {std_baseline_error:.4f}")
print(f"Win-rate over baseline:    {win_rate:.3f}")
