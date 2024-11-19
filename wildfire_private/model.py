# MODELO 3
import torch.nn as nn


import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, prob):
        super(MLPBlock, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=in_features)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(p=prob)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MLPphi(nn.Module):
    def __init__(self, hidden, features, prob):
        super(MLPphi, self).__init__()
        self.features = features
        self.hidden = hidden

        self.block1 = MLPBlock(in_features=features, out_features=hidden, prob=prob)
        self.block2 = MLPBlock(in_features=hidden, out_features=hidden, prob=prob)
        self.block3 = MLPBlock(in_features=hidden, out_features=hidden, prob=prob)

    def forward(self, u):
        # Input u: (batch, neighbors, features)
        batch_size, num_neighbors, _ = u.shape
        u_flat = u.view(-1, self.features)  # (batch * neighbors, features)

        # Passa pelos blocos com nomes descritivos
        output_block_1 = self.block1(u_flat)
        output_block_2 = self.block2(output_block_1)
        output_block_3 = self.block3(output_block_1 + output_block_2)

        final_output = output_block_1 + output_block_2 + output_block_3

        # Reformatar para (batch, neighbors, hidden)
        final_output = final_output.view(batch_size, num_neighbors, self.hidden)

        return final_output


##########################################################
#  Em construção
##########################################################


class MLPomega(nn.Module):
    def __init__(self, hidden, features, prob):
        super(MLPomega, self).__init__()
        self.features = features
        self.hidden = hidden

        self.layers = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.Linear(features, hidden, bias=False),
            nn.Tanh(),
            nn.Dropout(p=prob),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden, bias=False),
            nn.Tanh(),
            nn.Dropout(p=prob),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden, bias=False),
            nn.Softmax(dim=1),  # Attention weights
        )

    def forward(self, x, mask):
        x = self.layers(x)
        x = x.masked_fill(mask == 0, -float("inf"))
        return x


class MLPtheta(nn.Module):
    def __init__(self, hidden, prob):
        super(MLPtheta, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Dropout(p=prob),
            nn.Linear(hidden, hidden, bias=False),
            nn.Tanh(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(p=prob),
            nn.Linear(hidden, hidden, bias=False),
            nn.Tanh(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(p=prob),
            nn.Linear(hidden, hidden, bias=False),
            nn.Tanh(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(p=prob),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.layers(x)


class SpatialRegressor3(nn.Module):
    def __init__(self, hidden=32, features=4, prob=0.5):
        super(SpatialRegressor3, self).__init__()
        self.mlp_phi = MLPphi(hidden, features, prob)
        self.mlp_omega = MLPomega(hidden, features, prob)
        self.mlp_theta = MLPtheta(hidden, prob)

    def forward(self, u, mask):
        batch_size = u.shape[0]
        u = u.view(-1, self.mlp_phi.features)  # Flatten neighbors dimension

        # MLP_phi: Process features to latent space
        x_main = self.mlp_phi(u)
        x_main = x_main.view(
            batch_size, -1, self.mlp_phi.hidden
        )  # Reshape to batch x neighbors x hidden

        # MLP_omega: Compute attention weights
        x_attn = self.mlp_omega(u, mask)
        x_attn = x_attn.view(batch_size, -1, self.mlp_omega.hidden)

        # Attention mechanism
        x = x_main * x_attn  # Element-wise multiplication
        x = x.sum(dim=1)  # Aggregate over neighbors

        # MLP_theta: Final regression
        x = self.mlp_theta(x)
        return x
