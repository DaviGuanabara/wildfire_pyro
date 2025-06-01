from abc import abstractmethod
from typing import Callable, Union, Optional
import inspect
import sys
import time
from typing import Any, Dict, Optional, Tuple

from wildfire_pyro.wrappers.components.replay_buffer import ReplayBuffer
from wildfire_pyro.environments.base_environment import BaseEnvironment
from gymnasium import spaces
import numpy as np
import torch


from typing import TYPE_CHECKING, Optional, Union, Iterable, Dict, Any
from wildfire_pyro.common.logger import Logger, configure
from wildfire_pyro.wrappers.components.target_provider import BaseTargetProvider, InfoFieldTargetProvider
from .components.action_provider import BaseActionProvider

if TYPE_CHECKING:
    from wildfire_pyro.common.callbacks import (
        BaseCallback,
    )

import pathlib
import io
import zipfile
import pickle
import inspect



from wildfire_pyro.common.seed_manager import get_seed


def recursive_getattr(obj, attr, *default):
    """Helper to access nested attributes safely."""
    attributes = attr.split(".")
    for attribute in attributes:
        obj = getattr(obj, attribute, *default)
    return obj


def save_to_zip_file(
    path: Union[str, pathlib.Path, io.BufferedIOBase],
    data: Dict[str, Any],
    params: Dict[str, Any],
    pytorch_variables: Optional[Dict[str, Any]] = None,
) -> None:
    """Helper function to save an object to a compressed zip file."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        # Save non-PyTorch data
        archive.writestr("data.pkl", pickle.dumps(data))

        # Save PyTorch parameters
        with archive.open("params.pth", "w") as f:
            torch.save(params, f)

        # Save additional PyTorch variables
        if pytorch_variables is not None:
            with archive.open("pytorch_variables.pth", "w") as f:
                torch.save(pytorch_variables, f)

#TODO:
# target_extractor=lambda info: info["teacher_output"]["velocity"]
# Is it interesting to have a target extractor?
# It is a function that takes the info dictionary and returns the target value.
class BaseLearningManager:
    def __init__(
        self,
        environment: BaseEnvironment,
        neural_network: torch.nn.Module,
        runtime_parameters: Dict[str, Any],
        logging_parameters: Dict[str, Any],
        model_parameters: Dict[str, Any],
    ):
        """
        Initializes the learning manager.

        Args:
            environment (BaseEnvironment): Gymnasium environment instance.
            neural_network (torch.nn.Module): Neural network to be trained.
            parameters (Dict[str, Any]): Dictionary with training parameters.
        """
        
        # Target provider: default to InfoField
        self.target_provider: BaseTargetProvider = InfoFieldTargetProvider(
            "ground_truth")
        self.action_provider: BaseActionProvider = BaseActionProvider(
            provider=neural_network, device=runtime_parameters.get("device", "cpu"))
        
        self.environment = environment
        self.neural_network = neural_network
        # self.parameters = parameters
        self.device = runtime_parameters.get("device", "cpu")
        self.verbose = runtime_parameters.get("verbose", 1)
        self.seed = runtime_parameters.get("seed", 42)

        # self.tensorboard_log = parameters.get("tensorboard_log", None)

        self.log_path = logging_parameters.get("log_path")
        self.format_strings = logging_parameters.get("format_strings")
        # self.tensorboard_log = self.log_path

        self.batch_size = model_parameters.get("batch_size", 64)
        self.rollout_size = model_parameters.get("rollout_size", self.batch_size)
        self.lr = model_parameters.get("lr", 1e-3)

        # Initialize the environment state

        init_seed = get_seed("BaseLearningManager/init")
        self._last_obs, self._last_info = self.environment.reset(seed=init_seed)

        self.num_timesteps = 0
        self._total_timesteps = 0
        self.loss = np.inf

        obs_shape = environment.observation_space.shape
        assert obs_shape is not None, "observation_shape must not be None"

        # Experience replay buffer
        self.buffer = ReplayBuffer(
            max_size=self.batch_size,
            observation_shape=obs_shape,
            action_shape=(
                environment.action_space.shape
                if isinstance(environment.action_space, spaces.Box)
                else (1,)
            ),
            device=self.device,
        )

        self._custom_logger = False
        self.evaluation_metrics: Optional[Dict[str, Any]] = None


        
        self.lr_fn = LearningRateSchedulerWrapper(
            model_parameters.get("lr", 1e-3))

        self.optimizer = torch.optim.Adam(
            self.neural_network.parameters(),
            lr=self.lr_fn(step=0, total_steps=1, loss=None),
        )

        self.loss_func = torch.nn.MSELoss()

    def _update_learning_rate(
        self,
        **kwargs
    ) -> float:
        """
        Updates the learning rate based on training dynamics.

        Args:
            current_step: Current training step.
            total_steps: Total number of steps (for progress computation).
            loss: Most recent training loss.
            **kwargs: Additional context (e.g., evaluation_metrics).

        Returns:
            float: New learning rate.
        """

        current_step = self.num_timesteps
        total_steps=self._total_timesteps
        evaluation_metrics = self.evaluation_metrics or {}
        loss = self.loss  # Use the most recent loss


        new_lr = self.lr_fn(
            step=current_step,
            total_steps=total_steps,
            loss=loss,
            evaluation_metrics = evaluation_metrics,
            **kwargs  # pass other contextual signals like evaluation_metrics
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        if hasattr(self, "logger"):
            self.logger.record("train/lr", new_lr, exclude="stdout")

        if self.verbose > 0:
            print(
                f"[Info] Updated learning rate to {new_lr:.6f} at step {current_step}.")

        return new_lr
    

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
        from wildfire_pyro.common.callbacks import (
            BaseCallback,
            CallbackList,
            ConvertCallback,
            ProgressBarCallback,
            NoneCallback,
        )

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

    # TODO: Injetar aqui o action provider
    # PQ ELE vai dizer quem é o action source
    # o aluno ou o professor.
    
    def collect_rollouts(
        self,
        n_rollout_steps: int,
        callback: "BaseCallback",
    ) -> bool:
        """
        Collects rollouts from the environment and stores them in the buffer.

        Args:
            neural_network (torch.nn.Module): Neural network for action prediction.
            n_rollout_steps (int): Number of steps to collect.
        """
        callback.on_rollout_start()

        # Ensure we start with a valid observation
        obs, info = self.environment.reset(seed=self._generate_rollout_seed())

        for step in range(n_rollout_steps):
            
            action = self.action_provider.get_action(obs=obs)
            target = self.target_provider.get_target(info=info)

            if target is None:
                print(f"[Warning] Missing 'target'. Ending rollout.")
                break

            self.buffer.add(obs, action, target)

            obs, reward, terminated, truncated, info = self.environment.step(action)
            self.num_timesteps += 1

            callback.update_locals(locals())
            if not callback.on_step():
                print("[Info] Callback requested early stopping.")
                return False

            if terminated or truncated:
                obs, info = self.environment.reset(seed=self._generate_rollout_seed())

        callback.on_rollout_end()
        return True

    def _generate_rollout_seed(self) -> int:
        return get_seed(f"rollout_t{self.num_timesteps}")

    def _setup_learn(
        self,
        total_timesteps: int,
        reset_num_timesteps: bool = True,
        callback=None,
        tb_log_name="run",
        progress_bar: bool = False,
    ):
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
            self._last_obs, self._last_info = self.environment.reset(
                seed=self._generate_rollout_seed()
            )
        else:
            total_timesteps += self.num_timesteps

        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Configure logger
        if not self._custom_logger:
            self.logger = configure(self.log_path, self.format_strings)


        # Initialize callback
        callback = self._init_callback(callback, progress_bar)
        return total_timesteps, callback

    def learn(self, total_timesteps: int, callback=None, progress_bar: bool = False):
        """
        Main learning loop.
        Alternates between collecting rollouts and training the neural network.
        """
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback=callback, progress_bar=progress_bar
        )

        callback.on_training_start(locals(), globals())

        steps_completed = 0
        while steps_completed < total_timesteps:
            rollout_steps = min(self.batch_size, total_timesteps - steps_completed)

            continue_training = self.collect_rollouts(n_rollout_steps=rollout_steps, callback=callback)

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
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)

        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")

        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        self.logger.dump(step=self.num_timesteps)

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: Path to save the RL agent.
        :param exclude: Parameters to exclude from saving.
        :param include: Parameters to include (even if normally excluded).
        """
        path = pathlib.Path(path) if isinstance(path, str) else path

        # Copy instance attributes
        data = self.__dict__.copy()

        # Default parameters to exclude from saving
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)

        exclude.update(self._excluded_save_params())

        # Ensure explicitly included parameters are not excluded
        if include is not None:
            exclude.difference_update(include)

        # Handle PyTorch parameters
        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        pytorch_variables = {
            name: recursive_getattr(self, name) for name in torch_variable_names
        }

        # Remove excluded attributes
        for param_name in exclude:
            data.pop(param_name, None)

        # Get state dictionaries
        params_to_save = self.get_parameters()

        # Save everything to a zip file
        save_to_zip_file(
            path, data=data, params=params_to_save, pytorch_variables=pytorch_variables
        )

    def _excluded_save_params(self) -> set:
        """
        Returns a set of default attributes that should not be saved.
        """
        return {"logger", "environment", "buffer", "_last_obs", "_last_info"}

    def _get_torch_save_params(self):
        """
        Returns a tuple with:
        - List of names of state dictionaries to save.
        - List of names of PyTorch tensors/variables to save.
        """
        return ["neural_network.state_dict"], ["neural_network"]

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve model parameters.
        """
        return {"neural_network": self.neural_network.state_dict()}

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    def update_logger(
        self, folder: Optional[str] = None, format_strings: Optional[list[str]] = None
    ):
        self._logger = configure(folder=folder, format_strings=format_strings)
        self._custom_logger = True

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Any]:
        """
        Predict actions from observations.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def update_eval_metrics(
        self,
        evaluation_metrics: Dict[str, Any],
    ):
        """
        Receives evaluation metrics (e.g., from callbacks) to inform learning dynamics,
        such as adaptive learning rate schedulers.
        """
        self.evaluation_metrics = evaluation_metrics
        


class LearningRateSchedulerWrapper:
    """
    General-purpose wrapper for learning rate schedules.

    Accepts a float, a callable with any combination of (step, progress_remaining, loss),
    and handles dispatching based on the declared parameters.

    Example usage:
        scheduler = LearningRateSchedulerWrapper(
            lambda step, loss: 0.001 if loss > 0.05 else 0.0005)
        lr = scheduler(step=100, loss=0.04)
    """

    def __init__(self, lr: Union[float, Callable]):
        self.lr = lr

    def __call__(
        self,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        **kwargs: Any,
    ) -> float:
        
        if self.lr is None:
            raise ValueError(
                "Learning rate scheduler is None. Check model_parameters['lr']")

        if total_steps is not None and step is not None:
            progress_remaining = 1.0 - step / max(total_steps, 1)
        else:
            progress_remaining = None

        if callable(self.lr):
            return self._dispatch(
                self.lr,
                step=step,
                progress_remaining=progress_remaining,
                **kwargs
            )
        else:
            return float(self.lr)

    def _dispatch(self, fn, **context) -> float:
        sig = inspect.signature(fn)
        kwargs = {
            name: value
            for name, value in context.items()
            if name in sig.parameters and value is not None
        }
        return fn(**kwargs)
