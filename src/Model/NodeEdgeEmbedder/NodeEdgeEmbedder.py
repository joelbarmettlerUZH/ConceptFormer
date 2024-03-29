import logging

from src.Model.Embedder import Embedder
from src.Model.NodeEdgeEmbedder.EdgeNetwork import EdgeNetwork
from src.Model.NodeEdgeEmbedder.NodeNetwork import NodeNetwork
from src.Model.NodeEdgeEmbedder.AttentionAggregator import AttentionAggregator


class NodeEdgeEmbedder(Embedder):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation='gelu'):
        super(NodeEdgeEmbedder, self).__init__(node_dim, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation)
        self.edge_network = EdgeNetwork(edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation)
        self.node_network = NodeNetwork(node_dim, edge_dim, hidden_dim, output_dim, num_layers, activation)
        self.aggregator = AttentionAggregator(node_dim, edge_dim)

    def forward(self, central_node_features, neighbor_node_features, edge_features):
        """
        :param central_node_features: Tensor(batch_size, 1, embedding_size)
        :param neighbor_node_features: Tensor(batch_size, num_neighbors, embedding_size)
        :param edge_features: Tensor(batch_size, num_neighbors, embedding_size)
        :return: Tensor(batch_size, 1, embedding_size)
        """
        logging.debug(f"NodeEdgeEmbedder.central_node_features: {central_node_features.shape}")
        logging.debug(f"NodeEdgeEmbedder.neighbor_node_features: {neighbor_node_features.shape}")
        logging.debug(f"NodeEdgeEmbedder.edge_features: {edge_features.shape}")

        central_node_features = central_node_features.squeeze(1)

        edge_features_transformed = self.edge_network(edge_features)
        logging.debug(f"NodeEdgeEmbedder.edge_features_transformed: {edge_features_transformed.shape}")

        aggregated_features = self.aggregator(neighbor_node_features, edge_features_transformed)
        logging.debug(f"NodeEdgeEmbedder.aggregated_features: {aggregated_features.shape}")

        updated_central_node_features = self.node_network(central_node_features, aggregated_features)
        logging.debug(f"NodeEdgeEmbedder.updated_central_node_features: {updated_central_node_features.shape}")

        return updated_central_node_features.unsqueeze(1)
