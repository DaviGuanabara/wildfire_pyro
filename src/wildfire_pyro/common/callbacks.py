# ========================================================================================
# This code includes portions adapted from Stable-Baselines3 (SB3) under the MIT License.
# Original source: https://github.com/DLR-RM/stable-baselines3
#
# The MIT License (MIT)
#
# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# ========================================================================================

from wildfire_pyro.common.baselines.BaselineStrategy import BaselineStrategy
from wildfire_pyro.common.baselines.MeanNeighborBaseline import MeanNeighborBaseline
from dataclasses import asdict

from wildfire_pyro.common.evaluator import BootstrapEvaluator
from .logger import Logger
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Optional, List
from abc import ABC, abstractmethod
import warnings
import os
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.base_learning_manager import BaseLearningManager
import torch

from wildfire_pyro.common.messages import EvaluationMetrics
from wildfire_pyro.common.seed_manager import get_seed_manager, get_global_seed

import csv

try:
    from tqdm import TqdmExperimentalWarning

    # Remove experimental warning
    warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    from tqdm.rich import tqdm
except ImportError:
    tqdm = None  # Fallback when tqdm is not available


if TYPE_CHECKING:
    from wildfire_pyro.wrappers.base_learning_manager import BaseLearningManager


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    learner: "BaseLearningManager"

    def __init__(self, verbose: int = 0):
        super().__init__()
        self.n_calls = 0  # Number of times the callback was called
        self.num_timesteps = 0  # Total timesteps processed
        self.verbose = verbose
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        self.parent: Optional[BaseCallback] = None  # Optional parent callback

    @property
    def logger(self) -> Logger:
        return self.learner.logger

    def init_callback(self, learner: "BaseLearningManager") -> None:
        """
        Initialize the callback by linking it to the Learning Manager.
        """
        self.learner = learner
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        self.locals = locals_
        self.globals = globals_
        self.num_timesteps = self.learner.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    @abstractmethod
    def _on_step(self) -> bool:
        """
        Called at every step. Returning False stops training.
        """
        return True

    def on_step(self) -> bool:
        self.n_calls += 1
        self.num_timesteps = self.learner.num_timesteps
        return self._on_step()

    def on_training_end(self) -> None:
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: dict[str, Any]) -> None:
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        pass


class NoneCallback(BaseCallback):

    def __init__(
        self,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], bool]] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        Called at every step. Returning False stops training.
        """
        return True


class CallbackList(BaseCallback):
    """
    Handles multiple callbacks in sequence.

    :param callbacks: List of callback objects.
    """

    def __init__(self, callbacks: list[BaseCallback]):
        super().__init__()
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.learner)

            if hasattr(self, "parent") and self.parent is not None:
                callback.parent = self.parent  # Inherit parent if applicable

            

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        return all(callback.on_step() for callback in self.callbacks)

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        for callback in self.callbacks:
            callback.update_locals(locals_)


class ConvertCallback(BaseCallback):
    """
    Converts a function into a callback.

    :param callback: Function that takes (locals, globals) as arguments.
    """

    def __init__(
        self,
        callback: Optional[Callable[[dict[str, Any], dict[str, Any]], bool]],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None and callable(self.callback):
            return self.callback(self.locals, self.globals)
        return True


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints periodically.

    :param save_freq: Save every `save_freq` steps.
    :param save_path: Directory to save checkpoints.
    :param name_prefix: Prefix for the saved files.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "learner",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            self.learner.save(model_path)
            if self.verbose >= 1:
                print(f"Saved checkpoint at {model_path}")
        return True


class ProgressBarCallback(BaseCallback):
    """
    Shows a progress bar for training.
    """

    def __init__(self) -> None:
        super().__init__()
        if tqdm is None:
            raise ImportError("Install tqdm and rich for progress bars.")

    def _on_training_start(self) -> None:
        if tqdm is None:
            raise ImportError("Install tqdm and rich for progress bars.")
        else:
            self.pbar = tqdm(
                total=self.locals["total_timesteps"] - self.learner.num_timesteps
            )

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self) -> None:
        self.pbar.close()


class EventCallback(BaseCallback):

    def __init__(self, callback: Optional[BaseCallback] = None, verbose: int = 0):
        super(EventCallback, self).__init__(verbose=verbose)
        self.callback = callback
        # Give access to the parent
        if self.callback is not None:
            self.callback.parent = self

    def init_callback(self, learner) -> None:
        super(EventCallback, self).init_callback(learner)
        if self.callback is not None:
            self.callback.init_callback(self.learner)

    def _on_training_start(self) -> None:
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        if self.callback is not None:
            return self.callback.on_step()
        return True

    def _on_step(self) -> bool:
        #return True
        # TODO: FIZ ESSA MUDANÇA. TALVEZ DEVA VOLTAR ATRÁS.
        return self._on_event()


class BootstrapEvaluationCallback(EventCallback):
    def __init__(
        self,
        evaluation_environment: BaseEnvironment,
        error_function: Optional[Callable[[Any, Any], Any]] = None,
        n_eval: int = 5,
        n_bootstrap: int = 4,
        eval_freq: int = 1000,  # steps per evaluation (SB3-style)
        best_model_save_path: Optional[str] = None,
        verbose: int = False,
        seed: int = 42,
    ):
        """
        Callback to evaluate the learner using bootstrap sampling.
        :param evaluation_environment: Environment used for evaluation.
        :param error_function: Function to compute the error between predictions and ground truth.
        :param n_eval int: Number of evaluation runs.
        
            Number of independent bootstrap evaluations per prediction point.
            Each evaluation samples 'n_bootstrap' neighbors randomly.
        :param n_bootstrap: Number of bootstrap samples to use for evaluation.
        :param eval_freq: Frequency of evaluations in training steps.
        :param best_model_save_path: Path to save the best model based on evaluation error.
        :param verbose: Verbosity level (0 = no output, 1 = info, 2 = debug).
        :param seed: Seed for random number generation.

        
        """

        super().__init__(verbose=verbose)
        self.evaluation_environment: BaseEnvironment = evaluation_environment
        self.error_function = error_function if error_function else torch.nn.MSELoss()
        self.n_eval = n_eval
        self.eval_freq = eval_freq
        self.best_batch_error = np.inf  # Minimize error (not "loss")
        self.n_bootstrap = n_bootstrap

        self.best_model_save_path = best_model_save_path

        self.comparison_history: List[np.float32] = []
        self.rolling_window_size = 100
        self.seed = seed

        self._initial_log_done = False


    def _init_callback(self) -> None:
        super()._init_callback()

        if not self._initial_log_done:
            #self.logger.record("run/n_eval", self.n_eval)
            #self.logger.record("run/n_bootstrap", self.n_bootstrap)
            self.logger.dump(0)

    def _evaluate_learner(self) -> EvaluationMetrics:
        evaluator = BootstrapEvaluator(
            environment=self.evaluation_environment,
            learner=self.learner,
            n_eval=self.n_eval,
            n_bootstrap=self.n_bootstrap,
            error_function=self.error_function,
            seed=self.seed,
        )
        return evaluator.evaluate()

    def _on_step(self) -> bool:
        """Runs evaluation and logs results."""
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        results: EvaluationMetrics = self._evaluate_learner()

        if hasattr(self, "learner") and hasattr(self.learner, "update_eval_metrics"):
            self.learner.update_eval_metrics(results) # type: ignore

        self._log_to_logger(results)

        self._save_best_model(results.model_error)
        return True

    def _log_to_logger(self, results: EvaluationMetrics):

        # --- baseline (normalizado) ---
        self.logger.record(
            "val/baseline_loss_bootstrap",
            results.baseline_error,
            exclude=("tensorboard",)
        )
        self.logger.record(
            "val/baseline_loss_bootstrap_std",
            results.baseline_std,
            exclude=("tensorboard",),
        )

        # --- normalizado ---
        self.logger.record("val/loss_bootstrap", results.model_error)
        self.logger.record(
            "val/loss_bootstrap_std",
            results.model_std,
            exclude=("tensorboard",),
        )

        # --- raw (interpretável) ---
        if results.model_mae_raw is not None:
            self.logger.record("val/mae_raw_minutes", results.model_mae_raw)
            self.logger.record("val/rmse_raw_minutes", results.model_rmse_raw)

        if results.baseline_mae_raw is not None:
            self.logger.record("val/baseline_mae_raw_minutes",
                            results.baseline_mae_raw)
            self.logger.record("val/baseline_rmse_raw_minutes",
                            results.baseline_rmse_raw)

        if results.has_baseline():
            self.logger.record("eval/win_rate_over_baseline",
                            results.model_win_rate_over_baseline)

        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)


    def _save_best_model(self, model_error: float):

        if model_error < self.best_batch_error:
            if self.verbose > 0:
                self.logger.record(
                    "info", "New best model error achieved!", exclude=("csv", "tensorboard"))

            if self.best_model_save_path:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.learner.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_batch_error = model_error

class TrainLoggingCallback(BaseCallback):
    """
    Logs training loss periodically so it can be visualized alongside eval loss.

    :param log_freq: How often (in steps) to log the training loss.
    :param verbose: Verbosity level (0 = silent, 1 = info).
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.loss_history = []

    def _on_step(self) -> bool:
        """Called at every training step."""
        # Check if learner has a loss attribute updated by _train()
        if not hasattr(self.learner, "loss"):
            return True  # no loss yet

        loss_value = float(getattr(self.learner, "loss", np.nan))

        if not np.isfinite(loss_value):
            return True

        self.loss_history.append(loss_value)

        # Log periodically
        if self.n_calls % self.log_freq == 0:
            avg_loss = np.mean(self.loss_history[-self.log_freq:])
            std_loss = np.std(self.loss_history[-self.log_freq:])

            #raw_loss = self.learner.environment.to_raw_target(np.array([loss_value]))
            #self.learner.environment

            #self.logger.record("train/raw_loss", raw_loss)
            self.logger.record("train/loss", loss_value)
            self.logger.record("train/loss_avg", avg_loss)
            self.logger.record("train/loss_std", std_loss)
            

            


            

            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.dump(self.num_timesteps)

            #if self.verbose > 0:
            #   print(
            #        f"[TrainLoggingCallback] step={self.num_timesteps} | avg_loss={avg_loss:.4f}")
        return True
