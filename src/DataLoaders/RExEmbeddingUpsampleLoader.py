import random
from typing import Tuple, Dict

import networkx as nx
from torch.utils.data import Dataset, DataLoader

from src.Datasets.factory import rex_factory
from src.GraphAligner.BigGraphAligner import BigGraphAligner
from src.LLM.LLM import LLM


class RExEmbeddingUpsampleLoader(Dataset):

    def __init__(self, rex_data: Dataset, graphs: Dict[str, nx.DiGraph], graph_aligner: BigGraphAligner,
                 num_neighbors=100):
        self.graphs = graphs
        self.rex_data = rex_data
        self.num_neighbors = num_neighbors
        self.graph_aligner = graph_aligner

    def __len__(self):
        return len(self.rex_data)

    @staticmethod
    def random_fill(input_list, n):
        if n == 0:
            return []

        assert len(input_list) > 0, "Input list must have at least one element"

        if len(input_list) > n:
            return random.sample(input_list, n)

        output_list = input_list[:]
        random.shuffle(output_list)

        if len(input_list) == n:
            return output_list

        while len(output_list) < n:
            n_sample = min(len(input_list), n - len(output_list))
            output_list.extend(random.sample(input_list, n_sample))

        assert len(output_list) == n, f"Output list has incorrect length: {len(output_list)}"

        return output_list

    def __getitem__(self, idx):
        sentence = self.rex_data[idx]

        subject_id = sentence['subject']['id']
        subject_boundary_start, subject_boundary_end = sentence['subject']['boundaries']

        G = self.graphs[subject_id]

        central_node_embedding = self.graph_aligner.node_embedding_batch([subject_id])

        predicate_id = sentence['predicate']['id']

        object_id = sentence['object']['id']
        object_boundary_start, object_boundary_end = sentence['object']['boundaries']

        neighbour_ids, edge_ids = [], []
        for central_node_id, neighbour_node_id, edge in G.edges(data=True):
            if neighbour_ids != object_id:
                edge_ids.append(edge['id'])
                neighbour_ids.append(neighbour_node_id)

        # Graph has not enough nodes to fill batch, duplicates required, object node shall also be candidate for duplication
        if len(neighbour_ids) < (self.num_neighbors - 1):
            edge_ids.append(predicate_id)
            neighbour_ids.append(object_id)

        if self.num_neighbors > 1:
            node_edge_zip = self.random_fill(list(zip(neighbour_ids, edge_ids)), self.num_neighbors - 1)
            node_edge_zip.append((object_id, predicate_id))
            random.shuffle(node_edge_zip)
            reduced_neighbor_nodes, reduced_edge_attrs = zip(*node_edge_zip)
        else:
            reduced_neighbor_nodes = [object_id]
            reduced_edge_attrs = [predicate_id]

        assert len(reduced_neighbor_nodes) == self.num_neighbors, "Too many neighbors found"
        assert len(reduced_edge_attrs) == self.num_neighbors, "Too many edges found"

        node_embeddings = self.graph_aligner.node_embedding_batch(reduced_neighbor_nodes)
        edge_embeddings = self.graph_aligner.edge_embedding_batch(reduced_edge_attrs)

        assert central_node_embedding.shape == (
        1, self.graph_aligner.llm.embedding_length), "Central node dimension mismatch"
        assert node_embeddings.shape == (
        self.num_neighbors, self.graph_aligner.llm.embedding_length), "Neighbouring nodes dimension mismatch"
        assert edge_embeddings.shape == (
        self.num_neighbors, self.graph_aligner.llm.embedding_length), "Edges dimension mismatch"

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
    def from_dataset(train_dataset_name: str, graph_dataset_name: str, llm: LLM, num_neighbors: int) -> Tuple[
        "RExEmbeddingLoader", "RExEmbeddingLoader", "RExEmbeddingLoader"]:
        train_dataset, validation_dataset, test_dataset, graphs = rex_factory(train_dataset_name)

        graph_aligner = BigGraphAligner(llm, graphs, graph_dataset_name)

        train_loader = RExEmbeddingLoader(train_dataset, graphs, graph_aligner, num_neighbors=num_neighbors)
        validation_loader = RExEmbeddingLoader(validation_dataset, graphs, graph_aligner, num_neighbors=num_neighbors)
        test_loader = RExEmbeddingLoader(test_dataset, graphs, graph_aligner, num_neighbors=num_neighbors)
        return train_loader, validation_loader, test_loader

    @staticmethod
    def to_loader(train_dataset: "RExEmbeddingLoader", val_dataset: "RExEmbeddingLoader",
                  test_dataset: "RExEmbeddingLoader", batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        return train_loader, val_loader, test_loader
