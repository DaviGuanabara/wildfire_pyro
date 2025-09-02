import torch
import numpy as np
from typing import Union, Tuple, Optional



def predict_model(
    neural_network: torch.nn.Module,
    observation: np.ndarray,
    device: str,
    input_shape: Optional[Union[Tuple[int, ...], int]] = None,
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
        obs_tensor = torch.tensor(
            observation, dtype=torch.float32, device=device)

        # Try to infer if it's a batch
        if input_shape is None:
            is_batch = len(obs_tensor.shape) > 1  # assume batch if 2D+
        else:
            # Normalize input_shape
            if isinstance(input_shape, int):
                input_shape = (input_shape,)
            is_batch = len(obs_tensor.shape) == len(input_shape) + 1

        # Add batch dimension if not in batch format
        if not is_batch:
            # (1, num_neighbors, feature_dim)
            obs_tensor = obs_tensor.unsqueeze(0)

        # (batch_size, output_dim)
        action_tensor = neural_network(obs_tensor)
        action = action_tensor.cpu().numpy()

        # If input was not in batch, remove the batch dimension from the output
        if not is_batch:
            # (output_dim,)
            action = action.squeeze(0)

        # Return action(s) and an empty dictionary for additional information
        return action, {}
