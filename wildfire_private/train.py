import torch
import numpy as np
from typing import Any, Dict

from gymnasium import spaces
from wildfire_private.model import DeepSetAttentionNet

class EnvironmentManager:
    """
    Gerencia interações com o ambiente e coleta rollouts.
    """

    def __init__(self, env, buffer_size: int, device: str = "cpu"):
        """
        Inicializa o EnvironmentManager.

        Args:
            env: Instância do ambiente seguindo a API Gymnasium.
            buffer_size (int): Tamanho máximo do buffer para armazenar rollouts.
            device (str): Dispositivo para armazenar os tensores ("cpu" ou "cuda").
        """
        self.env = env
        self.device = device

        # Obtém as formas de observação e ação diretamente do ambiente
        self.obs_shape = self._get_obs_shape(env.observation_space)
        self.action_shape = self._get_action_shape(env.action_space)

        # Inicializa o buffer interno
        self.buffer_size = buffer_size
        self.observations = torch.zeros((buffer_size,) + self.obs_shape, device=device)
        self.actions = torch.zeros((buffer_size,) + self.action_shape, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.masks = torch.zeros(buffer_size, device=device)
        self.y_desired = torch.zeros((buffer_size, 1), device=device)

        self.pos = 0
        self.full = False

    @staticmethod
    def _get_obs_shape(observation_space):
        """
        Retorna a forma esperada da observação a partir do espaço de observação.
        """
        if isinstance(observation_space, spaces.Box):
            return observation_space.shape
        elif isinstance(observation_space, spaces.Discrete):
            return (1,)
        else:
            raise NotImplementedError(
                f"Espaço de observação {type(observation_space)} não suportado."
            )

    @staticmethod
    def _get_action_shape(action_space):
        """
        Retorna a forma esperada da ação a partir do espaço de ação.
        """
        if isinstance(action_space, spaces.Box):
            return action_space.shape
        elif isinstance(action_space, spaces.Discrete):
            return (1,)
        else:
            raise NotImplementedError(
                f"Espaço de ação {type(action_space)} não suportado."
            )

    def reset_buffer(self):
        """
        Reseta o buffer interno.
        """
        self.pos = 0
        self.full = False

    def add_to_buffer(self, obs, action, reward, y_desired, mask):
        """
        Adiciona um novo passo ao buffer.
        """
        self.observations[self.pos] = torch.tensor(obs, device=self.device)
        self.actions[self.pos] = torch.tensor(action, device=self.device)
        self.rewards[self.pos] = torch.tensor(reward, device=self.device)
        self.y_desired[self.pos] = torch.tensor(y_desired, device=self.device)
        self.masks[self.pos] = torch.tensor(mask, device=self.device)

        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def collect_rollouts(self, neural_network, n_rollout_steps: int):
        """
        Coleta rollouts do ambiente e armazena no buffer.
        """
        self.reset_buffer()
        obs, info = self.env.reset()

        for _ in range(n_rollout_steps):
            with torch.no_grad():
                action = neural_network.predict(torch.tensor(obs, device=self.device))

            new_obs, reward, done, truncated, info = self.env.step(action)
            y_desired = info.get("y_desired", None)

            if y_desired is None:
                print("[Warning] Missing y_desired. Ending rollout.")
                break

            self.add_to_buffer(obs, action, reward, y_desired, 1 - int(done))
            obs = new_obs

            if done or truncated:
                obs, info = self.env.reset()

    def get_batch(self, batch_size: int):
        """
        Retorna um batch de dados do buffer.
        """
        if batch_size > self.size():
            raise ValueError("Not enough data in buffer to fetch the batch.")

        start = (self.pos - batch_size) if self.full else 0
        end = self.pos
        indices = list(range(start, end))

        return (
            self.observations[indices],
            self.masks[indices],
            self.actions[indices],
            self.y_desired[indices],
        )

    def size(self):
        """
        Retorna o tamanho atual do buffer.
        """
        return self.buffer_size if self.full else self.pos


class DeepSetAttentionNetWrapper:
    """
    Wrapper para treinamento e inferência com DeepSetAttentionNet.
    """

    def __init__(
        self,
        neural_network: DeepSetAttentionNet,
        env,
        parameters: Dict[str, Any],
        n_steps: int = 2048,
        batch_size: int = 64,
    ):
        self.neural_network = neural_network
        self.env_manager = EnvironmentManager(
            env=env,
            buffer_size=n_steps,
            device=parameters.get("device", "cpu"),
        )
        self.parameters = parameters
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.device = parameters.get("device", "cpu")
        self.optimizer = torch.optim.Adam(neural_network.parameters(), lr=1e-3)
        self.loss_func = torch.nn.MSELoss()

    def train(self):
        """
        Treina o modelo usando os dados do buffer.
        """
        num_batches = max(1, self.env_manager.size() // self.batch_size)
        total_loss = 0.0

        for _ in range(num_batches):
            obs, mask, y_desired = self.env_manager.get_batch(self.batch_size)
            self.optimizer.zero_grad()
            y_pred = self.neural_network(obs, mask)
            loss = self.loss_func(y_pred, y_desired)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / num_batches

    def learn(self, total_steps):
        """
        Loop de aprendizado que alterna entre coleta de rollouts e treinamento.
        """
        for _ in range(total_steps // self.n_steps):
            self.env_manager.collect_rollouts(self.neural_network, self.n_steps)
            train_loss = self.train()
            print(f"[INFO] Train Loss: {train_loss:.4f}")

    def predict(self, obs, deterministic: bool = True):
        """
        Faz previsões usando o modelo treinado.
        """
        self.neural_network.eval()
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action = self.neural_network(obs_tensor).cpu().numpy()
        return action, None


def create_model(env, parameters: Dict[str, Any]) -> DeepSetAttentionNetWrapper:
    """
    Factory function to instantiate the DeepSetAttentionNet model and its wrapper.

    Args:
        env: Gymnasium environment instance.
        parameters: Dictionary containing model and training parameters.

    Returns:
        DeepSetAttentionNetWrapper: Configured model wrapped for training and inference.
    """
    # Obter dimensões de entrada (observação) do ambiente
    if isinstance(env.observation_space, spaces.Box):
        input_dim = env.observation_space.shape[-1]
    elif isinstance(env.observation_space, spaces.Discrete):
        input_dim = 1
    else:
        raise NotImplementedError(
            f"Unsupported observation space: {type(env.observation_space)}"
        )

    # Obter dimensões de saída (ação) do ambiente
    if isinstance(env.action_space, spaces.Box):
        output_dim = env.action_space.shape[-1]
    elif isinstance(env.action_space, spaces.Discrete):
        output_dim = env.action_space.n
    else:
        raise NotImplementedError(f"Unsupported action space: {type(env.action_space)}")

    # Configurar parâmetros adicionais
    parameters["input_dim"] = input_dim
    parameters["output_dim"] = output_dim

    # Instanciar o modelo com base nas dimensões do ambiente
    neural_network = DeepSetAttentionNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden=parameters.get("hidden", 32),
        prob=parameters.get("dropout_prob", 0.5),
    )

    # Retornar o wrapper configurado
    return DeepSetAttentionNetWrapper(neural_network, env, parameters)
