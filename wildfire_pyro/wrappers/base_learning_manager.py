import sys
import time
from typing import Any, Dict, Optional
from wildfire_pyro.wrappers.components.replay_buffer import ReplayBuffer
from wildfire_pyro.environments.base_environment import BaseEnvironment
import logging
from gymnasium import spaces
import numpy as np
import torch

import wildfire_pyro.common.logger as logger
from wildfire_pyro.common.utils import obs_as_tensor, safe_mean
from wildfire_pyro.common import utils


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wildfire_pyro.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback, NoneCallback



class BaseLearningManager:
    def __init__(self, environment: BaseEnvironment, neural_network: Any, parameters: Dict[str, Any]):
        """
        Initializes the learning manager.

        Args:
            environment (BaseEnvironment): Gymnasium environment instance.
            neural_network (Any): Neural network to be trained.
            parameters (Dict[str, Any]): Dictionary with training parameters.
        """
        self.environment = environment
        self.neural_network = neural_network
        self.parameters = parameters
        self.device = parameters.get("device", "cpu")
        self.verbose = parameters.get("verbose", 1)
        self.tensorboard_log = parameters.get("tensorboard_log", None)

        # Initialize logging
        log_dir = parameters.get("log_dir", None)
        self.logger = logger.configure(log_dir)

        # Initialize the environment state
        self._last_obs, self._last_info = self.environment.reset()
        self.num_timesteps = 0
        self._total_timesteps = 0

        # Experience replay buffer
        self.buffer = ReplayBuffer(
            max_size=parameters.get("batch_size", 64),
            observation_shape=environment.observation_space.shape,
            action_shape=(
                environment.action_space.shape if isinstance(
                    environment.action_space, spaces.Box) else (1,)
            ),
            device=self.device,
        )

        self._custom_logger = False

    def _init_callback(
        self,
        callback=None,
        progress_bar: bool = False,
    ):
        """
        Initializes the callback(s).

        :param callback: Callback(s) to be called at each step.
        :param progress_bar: Whether to display a progress bar.
        :return: A combined callback.
        """
        # Delayed import to avoid circular import
        from wildfire_pyro.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback, NoneCallback

        if callback is None:
            callback = NoneCallback()  # Default empty callback

        # Convert list of callbacks to a CallbackList
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar if needed
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def collect_rollouts(self, neural_network: torch.nn.Module, n_rollout_steps: int, callback: "BaseCallback") -> bool:
        """
        Collects rollouts from the environment and stores them in the buffer.

        Args:
            neural_network (torch.nn.Module): Neural network for action prediction.
            n_rollout_steps (int): Number of steps to collect.
        """
        callback.on_rollout_start()

        obs, info = self.environment.reset()  # Ensure we start with a valid observation

        for step in range(n_rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                y_pred: torch.Tensor = neural_network(obs_tensor)
                action: np.ndarray = y_pred.cpu().numpy().squeeze(0)

            ground_truth: Optional[float] = info.get("ground_truth", None)

            if ground_truth is None:
                print("[Warning] Missing ground_truth. Ending rollout.")
                break

            self.buffer.add(obs, action, ground_truth)

            obs, reward, terminated, truncated, info = self.environment.step(
                action)
            self.num_timesteps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                print("[Info] Callback requested early stopping.")
                return False

            if terminated or truncated:
                obs, info = self.environment.reset()

        callback.on_rollout_end()
        return True

    def _setup_learn(self, total_timesteps: int, reset_num_timesteps: bool = True, callback=None, tb_log_name="run",
                     progress_bar: bool = False):
        """
        Sets up the learning process before starting the training loop.

        Args:
            total_timesteps (int): Total number of timesteps for learning.
            reset_num_timesteps (bool): If True, resets the timestep counter.
        """
        self.start_time = time.time_ns()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
            self._last_obs, self._last_info = self.environment.reset()
        else:
            total_timesteps += self.num_timesteps

        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Configure logger
        if not self._custom_logger:
            self.logger = utils.configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )

        # Initialize callback
        callback = self._init_callback(callback, progress_bar)
        return total_timesteps, callback

    def learn(self, total_steps: int, callback=None, progress_bar: bool = False):
        """
        Main learning loop.
        Alternates between collecting rollouts and training the neural network.
        """
        total_timesteps, callback = self._setup_learn(
            total_steps, callback=callback, progress_bar=progress_bar)

        callback.on_training_start(locals(), globals())

        steps_completed = 0
        while steps_completed < total_timesteps:
            rollout_steps = min(self.parameters.get(
                "batch_size", 64), total_timesteps - steps_completed)

            continue_training = self.collect_rollouts(
                self.neural_network, n_rollout_steps=rollout_steps, callback=callback
            )

            if not continue_training:
                break

            self._train()
            steps_completed += rollout_steps

        callback.on_training_end()
        return self

    def _train(self) -> float:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def dump_logs(self, iteration: int = 0) -> None:
        """
        Writes logs.

        :param iteration: Current logging iteration.
        """
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int(
            (self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        if iteration > 0:
            self.logger.record("time/iterations", iteration,
                               exclude="tensorboard")

        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed",
                           int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps",
                           self.num_timesteps, exclude="tensorboard")
        self.logger.dump(step=self.num_timesteps)
