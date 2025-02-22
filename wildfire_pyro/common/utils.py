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

import os
import platform
import random
import re
import glob
import numpy as np
import torch as th
from itertools import zip_longest
from typing import Optional, Union, Iterable
from collections import deque

import cloudpickle
from wildfire_pyro.common.logger import Logger

# Check if tensorboard is available for PyTorch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """
    Seed random generators for reproducibility.

    :param seed: Random seed.
    :param using_cuda: Whether to apply CUDA-specific seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    if using_cuda:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the explained variance of predictions.

    :param y_pred: Predicted values.
    :param y_true: Ground truth values.
    :return: Explained variance score.
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate of an optimizer.

    :param optimizer: PyTorch optimizer.
    :param learning_rate: New learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def get_linear_fn(start: float, end: float, end_fraction: float):
    """
    Linear interpolation function.

    :param start: Starting value.
    :param end: Ending value.
    :param end_fraction: Fraction of training steps where `end` is reached.
    :return: Linear schedule function.
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_fn(val: float):
    """
    Returns a function that always outputs a constant value.

    :param val: Constant value.
    :return: Function that returns `val`.
    """

    def func(_):
        return val

    return func


def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.

    :param device: "auto", "cuda", or "cpu".
    :return: Selected PyTorch device.
    """
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    return th.device(device)


def get_latest_run_id(log_path: str = "", log_name: str = "") -> int:
    """
    Find the highest run ID in a log directory.

    :param log_path: Log directory.
    :param log_name: Experiment name.
    :return: Latest run ID.
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
        file_name = os.path.basename(path)
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit():
            max_run_id = max(max_run_id, int(ext))
    return max_run_id


def configure_logger(verbose: int = 0, tensorboard_log: Optional[str] = None, tb_log_name: str = "",
                     reset_num_timesteps: bool = True) -> Logger:
    """
    Set up logging.

    :param verbose: Logging verbosity.
    :param tensorboard_log: Tensorboard log directory.
    :param tb_log_name: Name for tensorboard logs.
    :param reset_num_timesteps: Whether to reset timestep count.
    :return: Logger instance.
    """
    save_path, format_strings = None, ["stdout"]

    if tensorboard_log is not None and SummaryWriter is None:
        raise ImportError(
            "TensorBoard logging requires `pip install tensorboard`.")

    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            latest_run_id -= 1
        save_path = os.path.join(
            tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        format_strings = [
            "stdout", "tensorboard"] if verbose >= 1 else ["tensorboard"]
    elif verbose == 0:
        format_strings = [""]

    return Logger(save_path, format_strings=format_strings)


def safe_mean(arr: Union[np.ndarray, list, deque]) -> float:
    """
    Compute the mean safely, returning NaN for empty lists.

    :param arr: Input array or list.
    :return: Mean value or NaN.
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))


def polyak_update(params: Iterable[th.Tensor], target_params: Iterable[th.Tensor], tau: float) -> None:
    """
    Perform Polyak averaging.

    :param params: Source parameters.
    :param target_params: Target parameters.
    :param tau: Polyak averaging coefficient (0 = no update, 1 = full copy).
    """
    with th.no_grad():
        for param, target_param in zip_longest(params, target_params):
            if param is None or target_param is None:
                raise ValueError(
                    "Mismatched parameter shapes in Polyak update.")
            target_param.data.mul_(1 - tau).add_(param.data, alpha=tau)


def obs_as_tensor(obs: Union[np.ndarray, dict[str, np.ndarray]], device: th.device) -> Union[th.Tensor, dict[str, th.Tensor]]:
    """
    Convert an observation to a PyTorch tensor.

    :param obs: Observation (array or dictionary).
    :param device: PyTorch device.
    :return: Converted tensor.
    """
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(value, device=device) for key, value in obs.items()}
    else:
        raise TypeError(f"Unsupported observation type {type(obs)}")


def get_system_info(print_info: bool = True) -> dict[str, str]:
    """
    Retrieve system and Python environment info.

    :param print_info: Whether to print info.
    :return: Dictionary of system information.
    """
    env_info = {
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "PyTorch": th.__version__,
        "GPU Enabled": str(th.cuda.is_available()),
        "Numpy": np.__version__,
        "Cloudpickle": cloudpickle.__version__,
    }

    if print_info:
        for key, value in env_info.items():
            print(f"- {key}: {value}")

    return env_info
