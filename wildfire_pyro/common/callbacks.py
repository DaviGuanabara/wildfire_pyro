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
