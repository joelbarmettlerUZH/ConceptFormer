import logging

from torch import nn, cat
import torch.nn.functional as F


class NodeNetwork(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers, activation='gelu'):
        super(NodeNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(node_dim + (edge_dim * 2), hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))    # BatchNorm after each linear layer
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim)) # BatchNorm after each linear layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation

    def forward(self, node_features, aggregated_features):
        """
        :param node_features: Tensor(batch_size, embedding_size)
        :param aggregated_features: Tensor(batch_size, 2*embedding_size)
        :return: Tensor(batch_size, embedding_size)
        """
        logging.debug(f"NodeNetwork.node_features: {node_features.shape}", )
        logging.debug(f"NodeNetwork.aggregated_features: {aggregated_features.shape}")
        x = cat([node_features, aggregated_features], dim=-1)
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
