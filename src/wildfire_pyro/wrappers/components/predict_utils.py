from gymnasium import spaces
import torch
import numpy as np
from typing import Union, Tuple, Optional


from typing import Dict, Tuple


def predict_model(
    neural_network: torch.nn.Module,
    observation: Dict[str, np.ndarray],
    device: str,
    observation_space: spaces.Dict,
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

        action_tensor = neural_network(obs_tensor)
        if isinstance(action_tensor, tuple):
            action_tensor = action_tensor[0]

        action = action_tensor.cpu().numpy()
        return action, {}


def to_obs_tensor(
    observation: Dict[str, np.ndarray],
    observation_space: spaces.Dict,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Converte observação (dict, lista de dicts ou array) em tensores com batch.
    Usa o observation_space para verificar consistência dinamicamente.
    """

    #print("observation type:", type(observation))
    #print("observation:", observation)
    #print("observation_space type:", type(observation_space))
    #print("observation_space:", observation_space)
    # Caso Dict space
    out = {}
    for k, sp in observation_space.spaces.items():
        t = torch.as_tensor(
            observation[k], dtype=torch.float32, device=device
        )

        # Ensure batch dimension
        if t.ndim == len(sp.shape):  # type: ignore
            t = t.unsqueeze(0)

        out[k] = t

    return out
