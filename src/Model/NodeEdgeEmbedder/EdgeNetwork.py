import logging

from torch import nn
import torch.nn.functional as F


class EdgeNetwork(nn.Module):
    def __init__(self, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation='gelu'):
        super(EdgeNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(edge_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(num_neighbors))    # BatchNorm after each linear layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(num_neighbors)) # BatchNorm after each linear layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation

    def forward(self, edge_features):
        """
        :param edge_features: Tensor(batch_size, num_neighbors, embedding_size)
        :return: Tensor(batch_size, num_neighbors, embedding_size)
        """
        logging.debug(f"EdgeNetwork.edge_features: {edge_features.shape}")
        x = edge_features
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation == 'gelu':
                    x = F.gelu(x)
                elif self.activation == 'relu':
                    x = F.relu(x)
                elif self.activation == 'leaky_relu':
                    x = F.leaky_relu(x)
        return x
