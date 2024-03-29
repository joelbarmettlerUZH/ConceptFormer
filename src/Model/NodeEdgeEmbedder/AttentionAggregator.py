import logging

from torch import nn, cat, mean, tanh
import torch.nn.functional as F


class AttentionAggregator(nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(AttentionAggregator, self).__init__()
        self.attention = nn.Linear(node_dim + edge_dim, 1)

    def forward(self, node_features, edge_features_transformed):
        """
        :param node_features: Tensor(batch_size, num_neighbors, embedding_size)
        :param edge_features_transformed: Tensor(batch_size, num_neighbors, embedding_size)
        :return: Tensor(batch_size, 2*embedding_size)
        """
        logging.debug(f"AttentionAggregator.node_features: {node_features.shape}")
        logging.debug(f"AttentionAggregator.edge_features_transformed: {edge_features_transformed.shape}")
        features = cat([node_features, edge_features_transformed], dim=-1)
        logging.debug(f"AttentionAggregator.features: {features.shape}")
        attention_weights = self.attention(tanh(features))
        logging.debug(f"AttentionAggregator.attention_weights: {attention_weights.shape}")
        attention_softmax = F.softmax(attention_weights, dim=1)
        logging.debug(f"AttentionAggregator.attention_softmax: {attention_softmax.shape}")
        weighted_features = attention_softmax * features
        logging.debug(f"AttentionAggregator.weighted_features: {weighted_features.shape}")
        aggregated_features = mean(weighted_features, dim=1)
        return aggregated_features

