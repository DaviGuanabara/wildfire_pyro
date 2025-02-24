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


from .logger import Logger
import numpy as np
import gymnasium as gym
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from abc import ABC, abstractmethod
import warnings
import os
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.base_learning_manager import BaseLearningManager
import torch


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # Handle cases where TensorBoard is missing

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
        self.parent = None  # Optional parent callback

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

    def on_training_start(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
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

    def __init__(self, callback: Optional[Callable[[dict[str, Any], dict[str, Any]], bool]] = None, verbose: int = 0):
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

    def __init__(self, callback: Optional[Callable[[dict[str, Any], dict[str, Any]], bool]], verbose: int = 0):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        if self.callback is not None:
            return self.callback(self.locals, self.globals)
        return True


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints periodically.

    :param save_freq: Save every `save_freq` steps.
    :param save_path: Directory to save checkpoints.
    :param name_prefix: Prefix for the saved files.
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "learner", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
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
        self.pbar = tqdm(
            total=self.locals["total_timesteps"] - self.learner.num_timesteps)

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
        if callback is not None:
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
        return True


class EvalCallback(EventCallback):
    """
    Evaluation Callback for supervised learning.

    :param eval_env: (BaseEnvironment) The environment used for validation.
    :param learner: (BaseLearningManager) The learning manager handling training.
    :param loss_function: (Callable) Function to compute loss between predictions and ground truth.
    :param n_eval_episodes: (int) Number of evaluation episodes.
    :param eval_freq: (int) Evaluate the model every `eval_freq` training steps.
    :param log_path: (Optional[str]) Path to save evaluation results.
    :param best_model_save_path: (Optional[str]) Path to save best model.
    :param tensorboard_log: (Optional[str]) Path for TensorBoard logging.
    :param verbose: (int) Verbosity level (0 = silent, 1 = print evaluation results).
    """

    def __init__(
        self,
        eval_env: BaseEnvironment,
        loss_function: Callable[[Any, Any], Any] = torch.nn.MSELoss(),  # Should be a function like `torch.nn.MSELoss()`
        n_eval_episodes: int = 5,
        eval_freq: int = 1000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.loss_function = loss_function
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.best_mean_loss = np.inf  # Minimize loss in supervised learning

        self.log_path = os.path.join(log_path, "evaluations") if log_path else None
        self.evaluations_results = []
        self.evaluations_timesteps = []

        self.tensorboard_log = tensorboard_log
        self.tb_writer = SummaryWriter(log_dir=tensorboard_log) if tensorboard_log and SummaryWriter else None

    def _init_callback(self) -> None:
        """Initializes callback and creates necessary directories."""
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    #TODO: Trocar _evaluate_model por _evaluate_learner
    def _evaluate_model(self) -> float:
        """
        Evaluates the model on the validation environment.

        :return: Mean loss across episodes.
        """
        total_loss = 0.0

        for _ in range(self.n_eval_episodes):
            
            # TODO: isso daqui tá errado!
            # o reset retorna uma tupla de obs e info. Em info, eu tenho que estrair o true_label (ground truth)
            raise ValueError('Corrigir isso aqui.')
            eval_reset = self.eval_env.reset()
            obs = eval_reset[0] if isinstance(eval_reset, tuple) else eval_reset
            true_label = eval_reset[1] if isinstance(eval_reset, tuple) and len(eval_reset) > 1 else None

            done = False
            while not done:
                pred = self.learner.predict(obs)

                if true_label is not None:
                    loss = self.loss_function(pred, true_label)  # Compute loss
                    total_loss += loss.item()  # Convert Tensor to float if using PyTorch

                step_result = self.eval_env.step(pred)
                obs = step_result[0] if isinstance(step_result, tuple) else step_result
                true_label = step_result[1] if isinstance(step_result, tuple) and len(step_result) > 1 else None
                done = step_result[2] if isinstance(step_result, tuple) and len(step_result) > 2 else False

        mean_loss = total_loss / self.n_eval_episodes
        return mean_loss

    def _on_step(self) -> bool:
        """Runs evaluation and logs results."""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            mean_loss = self._evaluate_model()

            if self.log_path:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(mean_loss)
                np.savez(self.log_path, timesteps=self.evaluations_timesteps, results=self.evaluations_results)

            if self.verbose > 0:
                print(f"[Evaluation] Num Timesteps: {self.num_timesteps} | Loss: {mean_loss:.4f}")

            if hasattr(self.learner, "logger") and isinstance(self.learner.logger, Logger):
                self.learner.logger.record("eval/loss", mean_loss)
                self.learner.logger.record("time/total_timesteps", self.num_timesteps)
                self.learner.logger.dump(self.num_timesteps)

            if self.tb_writer:
                self.tb_writer.add_scalar("eval/loss", mean_loss, self.num_timesteps)
                self.tb_writer.flush()

            if mean_loss < self.best_mean_loss:
                if self.verbose > 0:
                    print("New best loss achieved!")
                if self.best_model_save_path:
                    self.learner.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_loss = mean_loss

        return True