# MODELO 3
import torch.nn as nn


import torch.nn as nn


class MLPBlock(nn.Module):
    """
    Basic MLP block with batch normalization, linear layer, optional activation, and optional dropout regularization.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        prob (float): Dropout probability. Ignored if use_dropout is False.
        activation_function (callable, optional): Activation function applied after the linear layer.
            If None, no activation is applied. Default: None.
        use_dropout (bool): Whether to apply dropout regularization. Default: True.

    Attributes:
        batch_norm (nn.BatchNorm1d): Batch normalization for stable training.
        linear (nn.Linear): Linear transformation of input features.
        activation (callable): User-defined activation function.
        dropout (nn.Dropout or None): Dropout regularization to prevent overfitting. None if use_dropout is False.
    """

    def __init__(
        self,
        in_features,
        out_features,
        activation_function=None,
        use_dropout=False,
        prob=0.1,
    ):
        super(MLPBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=in_features)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation_function
        self.dropout = nn.Dropout(p=prob) if use_dropout else None

    def forward(self, x):
        """
        Performs the forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after batch normalization, linear transformation,
            activation (if applicable), and dropout (if applicable).
        """
        x = self.batch_norm(x)
        x = self.linear(x)
        x = self.activation(x) if self.activation is not None else x
        x = self.dropout(x) if self.dropout is not None else x

        return x


class MLPphi(nn.Module):
    """
    MLP responsible for aggregating information from neighbors.

    This module consists of three MLP blocks with residual connections between them,
    transforming the input data hierarchically.

    Args:
        hidden (int): Hidden layer size in each block.
        features (int): Number of input features.
        prob (float): Dropout probability.

    Attributes:
        phi_block_1 (MLPBlock): First MLP block.
        phi_block_2 (MLPBlock): Second MLP block.
        phi_block_3 (MLPBlock): Third MLP block.
    """

    def __init__(self, hidden, features, prob):
        super(MLPphi, self).__init__()
        self.features = features
        self.hidden = hidden

        self.phi_block_1 = MLPBlock(
            in_features=features,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.phi_block_2 = MLPBlock(
            in_features=hidden,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.phi_block_3 = MLPBlock(
            in_features=hidden,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )

    def forward(self, u):
        """
        Performs the forward pass of the MLPphi.

        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, num_neighbors, features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_neighbors, hidden),
            after the transformations of the blocks and residual summation.
        """
        batch_size, num_neighbors, _ = u.shape
        u_flat = u.view(-1, self.features)  # (batch * neighbors, features)

        input_block_1 = u_flat
        output_block_1 = self.phi_block_1(input_block_1)

        input_block_2 = output_block_1
        output_block_2 = self.phi_block_2(input_block_2)

        input_block_3 = output_block_2 + input_block_2
        output_block_3 = self.phi_block_3(input_block_3)

        final_output = output_block_3 + input_block_3
        final_output = final_output.view(batch_size, num_neighbors, self.hidden)

        return final_output


class MLPomega(nn.Module):
    """
    MLP responsible for computing attention weights.

    This module uses three MLP blocks to process input data and applies a softmax function
    to generate normalized attention weights.

    Args:
        hidden (int): Hidden layer size in each block.
        features (int): Number of input features.
        prob (float): Dropout probability.

    Attributes:
        omega_block_1 (MLPBlock): First MLP block.
        omega_block_2 (MLPBlock): Second MLP block.
        omega_block_3 (MLPBlock): Third MLP block.
        output_function (nn.Softmax): Softmax function for weight normalization.
    """

    def __init__(self, hidden, features, prob):
        super(MLPomega, self).__init__()
        self.features = features
        self.hidden = hidden

        self.omega_block_1 = MLPBlock(
            in_features=features,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.omega_block_2 = MLPBlock(
            in_features=hidden,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.omega_block_3 = MLPBlock(
            in_features=hidden,
            out_features=hidden,
            use_dropout=True,
            prob=prob,
        )

        self.output_function = nn.Softmax(dim=1)

    def forward(self, u, mask):
        """
        Performs the forward pass of the MLPomega.

        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, num_neighbors, features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_neighbors, hidden),
            containing normalized attention weights.
        """
        batch_size, num_neighbors, _ = u.shape
        u_flat = u.view(-1, self.features)  # (batch * neighbors, features)

        input_block_1 = u_flat
        output_block_1 = self.omega_block_1(input_block_1)

        input_block_2 = output_block_1
        output_block_2 = self.omega_block_2(input_block_2)

        input_block_3 = output_block_2 + input_block_2
        output_block_3 = self.omega_block_3(input_block_3)

        output_block_3_masked = output_block_3.masked_fill(mask == 0, -float("inf"))

        final_output = self.output_function(output_block_3_masked)
        final_output = final_output.view(batch_size, num_neighbors, self.hidden)

        return final_output


class MLPtheta(nn.Module):
    """
    MLP responsible for final regression.

    This module consists of four MLP blocks with residual connections, where the final block
    reduces the dimensionality to a single value for each instance.

    Args:
        hidden (int): Hidden layer size in each block.
        prob (float): Dropout probability.

    Attributes:
        theta_block_1 (MLPBlock): First MLP block.
        theta_block_2 (MLPBlock): Second MLP block.
        theta_block_3 (MLPBlock): Third MLP block.
        theta_block_4 (MLPBlock): Fourth MLP block for dimensionality reduction to a single output.
    """

    def __init__(self, hidden, features, prob):
        super(MLPtheta, self).__init__()
        self.features = features
        self.hidden = hidden

        self.theta_block_1 = MLPBlock(
            in_features=features,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.theta_block_2 = MLPBlock(
            in_features=hidden,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )
        self.theta_block_3 = MLPBlock(
            in_features=hidden,
            out_features=hidden,
            activation_function=nn.Tanh(),
            use_dropout=True,
            prob=prob,
        )

        self.theta_block_4 = MLPBlock(
            in_features=hidden,
            out_features=1,  # Reduces to a single output
            activation_function=None,
            use_dropout=False,
        )

        self.output_function = nn.Softmax(dim=1)

    def forward(self, u):
        """
        Performs the forward pass of the MLPtheta.

        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, hidden).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """

        batch_size, num_neighbors, _ = u.shape
        u_flat = u.view(-1, self.features)  # (batch * neighbors, features)

        input_block_1 = u_flat
        output_block_1 = self.theta_block_1(input_block_1)

        input_block_2 = output_block_1 + input_block_1
        output_block_2 = self.theta_block_2(input_block_2)

        input_block_3 = output_block_2 + input_block_2
        output_block_3 = self.theta_block_3(input_block_3)

        input_block_4 = output_block_3 + input_block_3
        output_block_4 = self.theta_block_4(input_block_4)

        final_output = output_block_4
        final_output = final_output.view(batch_size, num_neighbors, self.hidden)

        return final_output


class DeepSetAttentionNet(nn.Module):
    """
    Neural network model implementing a Deep Set Attention mechanism with three main components:
    MLP_phi, MLP_omega, and MLP_theta.

    This architecture is designed to process input data with a neighbor dimension, extract latent features
    using MLP_phi, compute attention weights using MLP_omega, and perform final regression using MLP_theta.
    It supports flexible input and output dimensions, making it suitable for tasks involving feature
    aggregation across neighbors.

    Args:
        input_dim (int): The number of input features per neighbor.
        output_dim (int): The number of output features from the model.
        hidden (int, optional): The size of the hidden layers for all MLP components. Default: 32.
        prob (float, optional): Dropout probability for regularization in MLP blocks. Default: 0.5.

    Attributes:
        mlp_phi (MLPphi): Module responsible for processing features into a latent space.
        mlp_omega (MLPomega): Module responsible for computing attention weights for each neighbor.
        mlp_theta (MLPtheta): Module responsible for final regression after feature aggregation.
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The dimensionality of the output features.
        hidden (int): The size of the hidden layers for all MLP components.
        prob (float): Dropout probability used for regularization.

    Methods:
        forward(u, mask):
            Processes input data through MLP_phi to extract latent features, applies MLP_omega to compute
            attention weights, aggregates features using weighted summation, and performs regression
            with MLP_theta. The mask is used to ignore padded elements during the computation.

    Forward Pass:
        1. Input tensor `u` of shape (batch_size, num_neighbors, input_dim) is reshaped to combine
        the batch and neighbor dimensions for processing.
        2. Latent features are extracted by MLP_phi from the flattened input.
        3. Attention weights are computed by MLP_omega and normalized using a mask.
        4. Element-wise multiplication combines latent features and attention weights.
        5. Weighted features are aggregated across the neighbor dimension using summation.
        6. Aggregated features are passed to MLP_theta for final regression, producing the output.

    Returns:
        torch.Tensor: Final output tensor of shape (batch_size, output_dim), representing the regression results.
    """

    def __init__(self, input_dim, output_dim, hidden=32, prob=0.5):
        super(DeepSetAttentionNet, self).__init__()

        self.mlp_phi = MLPphi(hidden=hidden, features=input_dim, prob=prob)
        self.mlp_omega = MLPomega(hidden=hidden, features=input_dim, prob=prob)
        self.mlp_theta = MLPtheta(hidden=hidden, features=hidden, prob=prob)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.prob = prob

    def forward(self, u, mask):
        """
        Forward pass of the DeepSetAttentionNet.

        Args:
            u (torch.Tensor): Input tensor of shape (batch_size, num_neighbors, input_dim).
            mask (torch.Tensor): Mask tensor of shape (batch_size, num_neighbors).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        batch_size, num_neighbors, _ = u.shape
        flattened_input = u.view(-1, self.input_dim)

        output_mlp_phi = self.mlp_phi(flattened_input)
        output_mlp_omega = self.mlp_omega(flattened_input, mask)

        weighted_features = output_mlp_phi * output_mlp_omega
        aggregated_features = weighted_features.sum(dim=1).view(batch_size, -1)

        output_mlp_theta = self.mlp_theta(aggregated_features)

        return output_mlp_theta
