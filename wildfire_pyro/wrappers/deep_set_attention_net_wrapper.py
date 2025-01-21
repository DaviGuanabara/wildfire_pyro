import torch
import numpy as np
from typing import Any, Dict, Tuple

from gymnasium import spaces
from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
import logging
from wildfire_pyro.environments.base_environment import BaseEnvironment

import torch
import numpy as np
from typing import Tuple, Optional


class ReplayBuffer:
    """
    Stores transitions (observation, action, ground_truth) for training.
    Implements a fixed-size buffer equal to batch_size.
    """

    def __init__(
        self,
        max_size: int,
        observation_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: str = "cpu",
    ):
        """
        Initializes the ReplayBuffer.

        Args:
            max_size (int): Number of transitions to store (typically equal to batch_size).
            observation_shape (Tuple[int, ...]): Shape of the observation space.
            action_shape (Tuple[int, ...]): Shape of the action space.
            device (str, optional): Device to store tensors. Defaults to "cpu".
        """
        self.device = device
        self.max_size = max_size
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        # Initialize tensors to store transitions
        self.observations = torch.zeros(
            (max_size,) + observation_shape, device=device)
        self.actions = torch.zeros((max_size,) + action_shape, device=device)
        self.ground_truth = torch.zeros((max_size, 1), device=device)

        self.position = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        ground_truth: float,
    ):
        """
        Adds a transition to the buffer.

        Args:
            obs (np.ndarray): Current observation.
            action (np.ndarray): Action taken.
            ground_truth (float): Optimal action associated.
        """
        if self.position < self.max_size:
            self.observations[self.position] = torch.tensor(
                obs, device=self.device, dtype=torch.float32
            )
            self.actions[self.position] = torch.tensor(
                action, device=self.device, dtype=torch.float32
            )
            self.ground_truth[self.position] = torch.tensor(
                ground_truth, device=self.device, dtype=torch.float32
            )
            self.position += 1

            if self.position == self.max_size:
                self.full = True
        else:
            raise RuntimeError(
                "ReplayBuffer is full. Call `reset` before adding more transitions."
            )

    def sample_batch(self) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Returns a batch of data from the buffer.

        Returns:
            Tuple containing observations, actions, and ground_truth.
        """
        if not self.full:
            raise ValueError("Buffer is not full yet.")
        # Return the entire buffer as the batch
        return (
            self.observations,
            self.actions,
            self.ground_truth,
        )

    def reset(self):
        """
        Resets the buffer by clearing all stored transitions.
        """
        self.position = 0
        self.full = False
        self.observations.zero_()
        self.actions.zero_()
        self.ground_truth.zero_()

    def is_full(self) -> bool:
        """
        Checks if the buffer is full.

        Returns:
            bool: True if buffer is full, False otherwise.
        """
        return self.full

    def size(self) -> int:
        """
        Returns the current number of transitions stored in the buffer.

        Returns:
            int: Number of transitions stored.
        """
        return self.max_size if self.full else self.position


class EnvDataCollector:
    """
    Interacts with the environment and collects data for the ReplayBuffer.
    """

    def __init__(
        self, environment: BaseEnvironment, buffer: ReplayBuffer, device: str = "cpu"
    ):
        """
        Initializes the EnvDataCollector.

        Args:
            environment (Any): Gymnasium environment instance.
            buffer (ReplayBuffer): Instance of ReplayBuffer to store data.
            device (str, optional): Device to store tensors. Defaults to "cpu".
        """
        self.environment: BaseEnvironment = environment
        self.buffer = buffer
        self.device = device

    def collect_rollouts(
            self, neural_network: torch.nn.Module, n_rollout_steps: int):
        """
        Collects rollouts from the environment and stores them in the buffer.

        Args:
            neural_network (torch.nn.Module): Neural network for action prediction.
            n_rollout_steps (int): Number of steps to collect.
        """
        obs, info = self.environment.reset()

        for _ in range(n_rollout_steps):
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    obs, device=self.device, dtype=torch.float32)
                y_pred: torch.Tensor = neural_network(
                    obs_tensor)  # (1, output_dim)
                action: np.ndarray = y_pred.cpu().numpy().squeeze(0)  # (output_dim,)

            ground_truth: Optional[float] = info.get("ground_truth", None)

            if ground_truth is None:
                print("[Warning] Missing ground_truth. Ending rollout.")
                break

            self.buffer.add(obs, action, ground_truth)

            obs, reward, done, truncated, info = self.environment.step(action)

            if done or truncated:
                obs, info = self.environment.reset()


class LearningManager:
    """
    Manages interactions with the environment, data collection, and neural network training.
    """

    def __init__(
        self,
        neural_network: Any,  # Replace with DeepSetAttentionNet if defined
        environment: BaseEnvironment,
        parameters: Dict[str, Any],
        batch_size: int = 64,
    ):
        """
        Initializes the LearningManager.

        Args:
            neural_network (Any): The neural network to be trained.
            environment (Any): Gymnasium environment instance.
            parameters (Dict[str, Any]): Configuration parameters.
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        self.neural_network = neural_network
        self.parameters = parameters
        self.batch_size = batch_size
        self.device = parameters.get("device", "cpu")
        self.environment: BaseEnvironment = environment

        # Initialize ReplayBuffer with max_size equal to batch_size
        self.buffer = ReplayBuffer(
            max_size=self.batch_size,
            observation_shape=environment.observation_space.shape,
            action_shape=(
                environment.action_space.shape
                if isinstance(environment.action_space, spaces.Box)
                else (1,)
            ),
            device=self.device,
        )

        # Initialize EnvDataCollector
        self.data_collector = EnvDataCollector(
            environment=environment, buffer=self.buffer, device=self.device
        )

        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.neural_network.parameters(), lr=parameters.get("lr", 1e-3)
        )
        self.loss_func = torch.nn.MSELoss()

    def train(self) -> float:
        """
        Trains the neural network using data from the buffer.

        Returns:
            float: Average training loss.
        """

        self.neural_network.train()
        buffer_size = self.buffer.size()
        if buffer_size < self.batch_size:
            print("[Warning] Not enough data in buffer to train. Skipping training.")
            return 0.0

        # Sample a batch (entire buffer)
        observations, actions, ground_truths = self.buffer.sample_batch()

        # (batch_size, num_neighbors, feature_dim)
        observations = observations.to(self.device)
        # (batch_size, 1)
        ground_truths = ground_truths.to(self.device)

        self.optimizer.zero_grad()

        # necessary to garanty that the actions are related to the current model.
        # (batch_size, output_dim)
        y_pred = self.neural_network(observations)

        loss = self.loss_func(y_pred, ground_truths)
        loss.backward()
        self.optimizer.step()

        average_loss: float = loss.item()
        print(f"[INFO] Train Loss: {average_loss:.4f}")

        return average_loss

    def learn(self, total_steps: int, rollout_steps: int):
        """
        Learning loop that alternates between collecting rollouts and training.

        Args:
            total_steps (int): Total number of steps for learning.
            rollout_steps (int): Number of steps per rollout.
        """
        steps_completed = 0
        while steps_completed < total_steps:
            current_rollout_steps = min(
                rollout_steps, total_steps - steps_completed)
            self.data_collector.collect_rollouts(
                neural_network=self.neural_network,
                n_rollout_steps=current_rollout_steps,
            )
            self.train()
            steps_completed += current_rollout_steps

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, Any]:
        """
        Makes predictions using the trained model.

        Args:
            obs (np.ndarray): Current observation. Can be a single observation or a batch.
            deterministic (bool, optional): If True, use deterministic prediction. Defaults to True.

        Returns:
            Tuple[np.ndarray, Any]: Predicted action(s) and additional information (empty dict).
        """

        # Set the network to evaluation mode
        self.neural_network.eval()
        with torch.no_grad():
            # Convert observation to tensor and move to the correct device
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device)
            expected_obs_shape = self.environment.observation_space.shape
            is_batch = len(obs_tensor.shape) == len(expected_obs_shape) + 1

            # Add batch dimension if not in batch format
            if not is_batch:
                # (1, num_neighbors, feature_dim)
                obs_tensor = obs_tensor.unsqueeze(0)

            # (batch_size, output_dim)
            action_tensor = self.neural_network(obs_tensor)
            action = action_tensor.cpu().numpy()

            # If input was not in batch, remove the batch dimension from the
            # output
            if not is_batch:
                # (output_dim,)
                action = action.squeeze(0)

        # Return action(s) and an empty dictionary for additional information
        return action, {}


def create_model(env: Any, parameters: Dict[str, Any]) -> LearningManager:
    """
    Factory function to instantiate the DeepSetAttentionNet model and its LearningManager.

    Args:
        env (Any): Gymnasium environment instance.
        parameters (Dict[str, Any]): Dictionary containing model and training parameters.

    Returns:
        LearningManager: Configured model manager for training and inference.
    """
    # Determine input dimensions from the environment's observation space
    if isinstance(env.observation_space, spaces.Box):
        input_dim = env.observation_space.shape[-1]
    elif isinstance(env.observation_space, spaces.Discrete):
        input_dim = 1
    else:
        raise NotImplementedError(
            f"Unsupported observation space: {type(env.observation_space)}"
        )

    # Determine output dimensions from the environment's action space
    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[-1]
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
    else:
        raise NotImplementedError(
            f"Unsupported action space: {type(env.action_space)}")

    # Configure additional parameters
    parameters["input_dim"] = input_dim
    parameters["output_dim"] = output_dim

    # Instantiate the neural network based on environment dimensions
    neural_network = DeepSetAttentionNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden=parameters.get("hidden", 32),
        prob=parameters.get("dropout_prob", 0.5),
    ).to(parameters.get("device", "cpu"))

    # Instantiate the LearningManager with buffer size equal to batch_size
    learning_manager = LearningManager(
        neural_network=neural_network,
        environment=env,
        parameters=parameters,
        batch_size=parameters.get("batch_size", 64),
    )

    return learning_manager
