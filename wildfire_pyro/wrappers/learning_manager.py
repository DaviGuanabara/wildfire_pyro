import torch
import numpy as np
from typing import Any, Dict, Tuple

from gymnasium import spaces

from wildfire_pyro.models.deep_set_attention_net import DeepSetAttentionNet
from wildfire_pyro.environments.base_environment import BaseEnvironment
from wildfire_pyro.wrappers.components.env_data_collector import EnvDataCollector
from wildfire_pyro.wrappers.components.replay_buffer import ReplayBuffer





class LearningManager:
    """
    Manages interactions with the environment, data collection, and neural network training.
    """

    def __init__(
        self,
        neural_network: Any, 
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

    def _train(self) -> float:
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
            self._train()
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



