from torch import nn


class Embedder(nn.Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation='gelu'):
        super(Embedder, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.activation = activation

    def forward(self, central_node_features, neighbor_node_features, edge_features):
        """
        :param central_node_features: Tensor(batch_size, 1, embedding_size)
        :param neighbor_node_features: Tensor(batch_size, num_neighbors, embedding_size)
        :param edge_features: Tensor(batch_size, num_neighbors, embedding_size)
        :return: Tensor(batch_size, embedding_size)
        """
        pass