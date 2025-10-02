from gymnasium import spaces
import torch
import numpy as np
from typing import Union, Tuple, Optional


def predict_model(
    neural_network: torch.nn.Module,
    observation: np.ndarray,
    device: str,
    observation_space: Optional[spaces.Space] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Performs forward pass with proper device handling and shape normalization.

    Args:
        neural_network (torch.nn.Module): The model to evaluate.
        observation (np.ndarray): Input observation, batched or not.
        device (str): Device where the model resides.
        input_shape (tuple or int, optional): Expected shape of a single sample.
            If None, inferred from observation shape.

    Returns:
        Tuple[np.ndarray, dict]: 
            - Model prediction(s), shape depends on input format.
            - Info dictionary (currently empty, but expandable).

    Notes:
        The returned info dictionary is reserved for future extensions such as:
        - 'logits': pre-softmax outputs
        - 'features': intermediate representations
        - 'attention': attention weights (if available)
        - 'confidence': optional uncertainty scores
    """

    neural_network.eval()

    with torch.no_grad():
        obs_tensor = to_obs_tensor(observation, observation_space, device)

        # 🔹 Caso Dict → repassa direto pro modelo
        if isinstance(obs_tensor, dict):
            action_tensor = neural_network(obs_tensor)
            if isinstance(action_tensor, tuple):
                action_tensor = action_tensor[0]  # pega só as ações
            action = action_tensor.cpu().numpy()

        else:  # 🔹 Caso Box → tensor único
            action_tensor = neural_network(obs_tensor)
            if isinstance(action_tensor, tuple):
                action_tensor = action_tensor[0]
            action = action_tensor.cpu().numpy()

        return action, {}


def to_obs_tensor(observation, observation_space, device):
    """
    Converte observação (dict ou array) em tensores com batch na frente.
    Usa o observation_space para checar consistência.
    """
    if isinstance(observation_space, spaces.Dict):
        out = {}
        for k, sp in observation_space.spaces.items():
            t = torch.as_tensor(
                observation[k], dtype=torch.float32, device=device)
            # Se t.ndim == len(sp.shape), falta batch
            if t.ndim == len(sp.shape):
                t = t.unsqueeze(0)
            # Se já veio batched, ok. Se veio errado, acusa
            elif t.ndim != len(sp.shape) + 1:
                raise ValueError(
                    f"Obs[{k}] com dims {t.shape}, esperado (B, {sp.shape}) ou {sp.shape}"
                )
            out[k] = t
        return out

    elif isinstance(observation_space, spaces.Box):
        t = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if t.ndim == len(observation_space.shape):
            t = t.unsqueeze(0)
        return t

    else:
        # fallback genérico (ex: observation_space=None)
        t = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        return t
