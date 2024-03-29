from typing import Tuple, Dict

import networkx as nx
from torch.utils.data import Dataset, DataLoader

from src.BatchSampler.DynamicNeighbourBatchSampler import DynamicNeighbourBatchSampler
from src.Datasets.factory import rex_factory
from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.LLM.LLM import LLM


class RExEmbeddingDynamicLoader(Dataset):

    def __init__(self, rex_data: Dataset, graphs: Dict[str, nx.DiGraph], graph_aligner: BigGraphAligner,
                 num_neighbors=100):
        self.graphs = graphs
        self.rex_data = rex_data
        self.num_neighbors = num_neighbors
        self.graph_aligner = graph_aligner

    def __len__(self):
        return len(self.rex_data)

    def num_neighbours_at_index(self, idx: int) -> int:
        sentence = self.rex_data[idx]
        subject_id = sentence['subject']['id']
        G = self.graphs[subject_id]

        num_neighbours = G.number_of_edges()
        return min(num_neighbours, self.num_neighbors)

    def __getitem__(self, idx):
        sentence = self.rex_data[idx]

        subject_id = sentence['subject']['id']
        subject_boundary_start, subject_boundary_end = sentence['subject']['boundaries']

        G = self.graphs[subject_id]

        central_node_embedding = self.graph_aligner.node_embedding_batch([subject_id])

        predicate_id = sentence['predicate']['id']

        object_id = sentence['object']['id']
        object_boundary_start, object_boundary_end = sentence['object']['boundaries']

        neighbour_ids, edge_ids, ranks = [], [], []
        for central_node_id, neighbour_node_id, edge in G.edges(data=True):
            edge_ids.append(edge['id'])
            neighbour_ids.append(neighbour_node_id)
            ranks.append(G.nodes[neighbour_node_id]['rank'])

        combined = zip(neighbour_ids, edge_ids, ranks)
        sorted_combined = sorted(combined, key=lambda x: x[2], reverse=True)
        neighbour_ids, edge_ids, _ = zip(*sorted_combined)
        neighbour_ids, edge_ids = list(neighbour_ids), list(edge_ids)

        if self.num_neighbors < len(neighbour_ids):
            neighbour_ids = neighbour_ids[:self.num_neighbors]
            edge_ids = edge_ids[:self.num_neighbors]

            if object_id not in neighbour_ids:
                neighbour_ids[-1] = object_id
                edge_ids[-1] = predicate_id

        node_embeddings = self.graph_aligner.node_embedding_batch(neighbour_ids)
        edge_embeddings = self.graph_aligner.edge_embedding_batch(edge_ids)

        return {
            'sentence': sentence['sentence'],

            'subject_id': sentence['subject']['id'],
            'subject_label': sentence['subject']['label'],
            'subject_rank': sentence['subject']['rank'],

            'predicate_id': sentence['predicate']['id'],
            'predicate_label': sentence['predicate']['label'],

            'object_id': sentence['object']['id'],
            'object_label': sentence['object']['label'],
            'object_rank': sentence['object']['rank'],

            'subject_boundary_start': subject_boundary_start,
            'subject_boundary_end': subject_boundary_end,

            'object_boundary_start': object_boundary_start,
            'object_boundary_end': object_boundary_end,

            "central_node_embedding": central_node_embedding,
            "node_embeddings": node_embeddings,
            "edge_embeddings": edge_embeddings,
        }

    @staticmethod
    def from_dataset(train_dataset_name: str, graph_dataset_name: str, llm: LLM, num_neighbors: int,
                     ignore_global_alignment=False) -> Tuple[
        "RExEmbeddingDynamicLoader", "RExEmbeddingDynamicLoader", "RExEmbeddingDynamicLoader"]:
        train_dataset, validation_dataset, test_dataset, graphs = rex_factory(train_dataset_name)

        graph_aligner = BigGraphAligner(llm, graphs, graph_dataset_name, use_untrained=ignore_global_alignment)

        train_loader = RExEmbeddingDynamicLoader(train_dataset, graphs, graph_aligner, num_neighbors=num_neighbors)
        validation_loader = RExEmbeddingDynamicLoader(validation_dataset, graphs, graph_aligner,
                                                      num_neighbors=num_neighbors)
        test_loader = RExEmbeddingDynamicLoader(test_dataset, graphs, graph_aligner, num_neighbors=num_neighbors)
        return train_loader, validation_loader, test_loader

    @staticmethod
    def to_loader(train_dataset: "RExEmbeddingDynamicLoader", val_dataset: "RExEmbeddingDynamicLoader",
                  test_dataset: "RExEmbeddingDynamicLoader", batch_size: int) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        train_sampler = DynamicNeighbourBatchSampler(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_sampler = DynamicNeighbourBatchSampler(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_sampler = DynamicNeighbourBatchSampler(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler)
        return train_loader, val_loader, test_loader
