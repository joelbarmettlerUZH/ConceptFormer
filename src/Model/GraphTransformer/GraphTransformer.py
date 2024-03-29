import torch
from torch import nn
import torch_geometric.nn as geom_nn


class GraphTransformerNet(nn.Module):
    def __init__(self, node_dim, edge_dim, in_channels, out_channels, heads=1, concat=True, beta=False, dropout=0, bias=True, root_weight=True):
        super(GraphTransformerNet, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.conv = geom_nn.TransformerConv(in_channels, out_channels, heads=heads, concat=concat, beta=beta, dropout=dropout, edge_dim=edge_dim, bias=bias, root_weight=root_weight)

    def forward(self, central_node_features, neighbor_node_features, edge_features):
        """
        :param central_node_features: Tensor(batch_size, 1, embedding_size)
        :param neighbor_node_features: Tensor(batch_size, num_neighbors, embedding_size)
        :param edge_features: Tensor(batch_size, num_neighbors, embedding_size)
        :return: Tensor(batch_size, 1, embedding_size)
        """
        batch_size, num_neighbors, _ = neighbor_node_features.size()
        x = torch.cat([central_node_features, neighbor_node_features], dim=1).view(-1, self.node_dim)
        edge_index = torch.cat([torch.zeros(batch_size, num_neighbors).long(),
                                torch.arange(1, num_neighbors + 1).repeat(batch_size, 1).long()], dim=0)
        edge_attr = edge_features.view(-1, self.node_dim)
        return self.conv(x, edge_index, edge_attr)
