import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Config.train_sentences_config import TrainSentencesConfig
from src.LLM.LLM import LLM
from src.Model.Embedder import Embedder


class GraphAttentionEmbedder(Embedder):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation='gelu', num_pseudo_words=1):
        super(GraphAttentionEmbedder, self).__init__(node_dim, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.d = num_pseudo_words

        # Define separate linear transformations for each head
        self.query_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])
        self.key_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])
        self.value_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])
        self.output_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])

        if self.activation == 'gelu':
            self.nonlinear_layer = nn.GELU
        elif self.activation == 'relu':
            self.nonlinear_layer = nn.ReLU
        elif self.activation == 'leaky_relu':
            self.nonlinear_layer = nn.LeakyReLU
        else:
            raise ValueError('Unknown activation')

        # Final neural network
        self.final_network = nn.ModuleList()
        self.final_network.append(nn.Linear(node_dim, hidden_dim))
        self.final_network.append(self.nonlinear_layer())
        # self.final_network.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.final_network.append(nn.Linear(hidden_dim, hidden_dim))
            self.final_network.append(self.nonlinear_layer())
            # self.final_network.append(nn.BatchNorm1d(hidden_dim))               # BatchNorm after each linear layer
        self.final_network.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, central_node_features, neighbor_node_features, edge_features):
        """
        :param central_node_features: Tensor(batch_size, 1, embedding_size)
        :param neighbor_node_features: Tensor(batch_size, num_neighbors, embedding_size)
        :param edge_features: Tensor(batch_size, num_neighbors, embedding_size)
        :return: Tensor(batch_size, 1, embedding_size)
        """
        logging.debug(f"GraphAttentionEmbedder.central_node_features: {central_node_features.shape}")
        logging.debug(f"GraphAttentionEmbedder.neighbor_node_features: {neighbor_node_features.shape}")
        logging.debug(f"GraphAttentionEmbedder.edge_features: {edge_features.shape}")

        batch_size, num_neighbors, _ = neighbor_node_features.size()

        multi_head_context = []

        for i in range(self.d):
            # Prepare Query, Key, Value for each head
            Q = self.query_transforms[i](central_node_features)  # (batch_size, 1, node_dim)
            K = self.key_transforms[i](neighbor_node_features) + edge_features  # (batch_size, num_neighbors, node_dim)
            V = self.value_transforms[i](neighbor_node_features)  # (batch_size, num_neighbors, node_dim)

            # Attention mechanism for each head
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.node_dim ** 0.5)
            logging.debug(f"GraphAttentionEmbedder.attention_scores: {attention_scores.shape}")

            attention_probs = F.softmax(attention_scores, dim=2)  # Softmax over neighbors
            logging.debug(f"GraphAttentionEmbedder.attention_probs: {attention_probs.shape}")

            # Weighted sum of values for each head
            context = torch.matmul(attention_probs, V)  # (batch_size, 1, node_dim)

            logging.debug(f"GraphAttentionEmbedder.context: {context.shape}")

            # Output transformation for each head
            head_output = self.output_transforms[i](context)
            logging.debug(f"GraphAttentionEmbedder.head_output: {head_output.shape}")

            output = head_output
            for module in self.final_network:
                output = module(output)

            logging.debug(f"GraphAttentionEmbedder.output: {output.shape}")

            multi_head_context.append(output)

        # Concatenate all head outputs
        multi_head_context = torch.cat(multi_head_context, dim=1)  # (batch_size, d, node_dim)

        logging.debug(f"GraphAttentionEmbedder.multi_head_context: {multi_head_context.shape}")

        return multi_head_context

    @staticmethod
    def from_config(config: TrainSentencesConfig, llm: LLM):
        return GraphAttentionEmbedder(
            node_dim=llm.embedding_length,
            edge_dim=llm.embedding_length,
            hidden_dim=int(llm.embedding_length * config.model_layer_width_multiplier),
            output_dim=llm.embedding_length,
            num_layers=config.model_layer_depth,
            num_neighbors=config.number_of_neighbors,
            activation=config.model_layer_activation,
            num_pseudo_words=config.num_pseudo_words,
        )