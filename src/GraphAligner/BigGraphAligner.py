import csv
import json
import os
from pathlib import Path
from typing import List, Dict

import h5py
import networkx as nx
import numpy as np
import torch
import wandb
from tqdm import tqdm
from torchbiggraph.converters.importers import TSVEdgelistReader
from torchbiggraph.graph_storages import FORMAT_VERSION_ATTR, FORMAT_VERSION
from torchbiggraph.config import ConfigSchema, EntitySchema, RelationSchema
from torchbiggraph.converters.importers import convert_input_data
from torchbiggraph.train import train
from torchbiggraph.util import SubprocessInitializer

from src.LLM.LLM import LLM


class BigGraphAligner:

    def __init__(self, llm: LLM, graphs: Dict[str, nx.DiGraph], dataset_name: str, epochs=1000, use_untrained=False):
        script_directory = Path(__file__).parent
        self.folder: Path = script_directory.parent.parent / "data" / "artifacts" / "BigGraphAlignment_v1" / dataset_name / llm.name
        os.makedirs(f'{self.folder}', exist_ok=True)
        self.llm: LLM = llm
        self.graphs: Dict[str, nx.DiGraph] = graphs
        self.dataset_name: str = dataset_name
        self.epochs: int = epochs
        self._use_untrained = use_untrained

        self.entity_index: Dict[str, np.array] = dict()   # Maps entity IDs to their embedding value
        self.relation_index: Dict[str, np.array] = dict() # Maps relation IDs to their embedding value

        self.prepare()
        self.train()
        self.build_index()


    def prepare(self):
        if os.path.isfile(f'{self.folder}/init/embeddings_entity_0.v0.h5'):
            print("Already prepared")
            return

        os.makedirs(f'{self.folder}/init', exist_ok=True)

        entities = dict()
        relations = dict()
        triples = set()

        # Open a file to write the edges
        for G in tqdm(self.graphs.values(), desc='Processing graphs', total=len(self.graphs)):
            for central_node_id, neighbour_node_id, edge in G.edges(data=True):
                assert central_node_id == G.graph['central_node'], "Graph has wrong format, expect all edges to be from central node"
                relation_id = edge['id']
                relation_label = edge['label']

                neighbour_node_label = G.nodes[neighbour_node_id]['label']
                central_node_label = G.nodes[central_node_id]['label']

                # Write the edge in the format required by PBG
                entities[neighbour_node_id] = neighbour_node_label
                entities[central_node_id] = central_node_label

                relations[relation_id] = relation_label

                triples.add((central_node_id, relation_id, neighbour_node_id))

        entity_list = [(key, value) for key, value in entities.items()]
        sorted_entity_list = sorted(entity_list, key=lambda x: int(x[0][1:]))

        relation_list = [(key, value) for key, value in relations.items()]
        sorted_relation_list = sorted(relation_list, key=lambda x: int(x[0][1:]))

        triples_list = list(triples)
        sorted_triples_list = sorted(triples_list, key=lambda x: (int(x[0][1:]), int(x[1][1:]), int(x[2][1:])))

        with open(f'{self.folder}/graph_edges.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['Subject', 'Predicate', 'Object'])
            writer.writerows(sorted_triples_list)

        # Write entities to a file
        with open(f'{self.folder}/entities.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Label'])
            for entity in tqdm(sorted_entity_list, desc='Writing entities'):
                writer.writerow(entity)

        # Write relations to a file
        with open(f'{self.folder}/relations.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Label'])
            for relation in tqdm(sorted_relation_list, desc='Writing relations'):
                writer.writerow(relation)

        cpu = torch.device("cpu")
        dataset = torch.zeros((len(sorted_entity_list), self.llm.embedding_length), dtype=torch.float32).to(cpu)
        for i, entity in tqdm(list(enumerate(sorted_entity_list)), desc='Embedding entities'):
            dataset[i, :] = self.llm.late_embedding(entity[1]).squeeze(0).squeeze(0).to(cpu)

        with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'w') as hf:
            hf.create_dataset("embeddings", data=dataset.cpu().numpy(), dtype=np.float32)
            hf.attrs[FORMAT_VERSION_ATTR] = FORMAT_VERSION

    def train(self):
        if self._use_untrained:
            print("Using untrained BigGraphAligner")
            return

        if os.path.isfile(f'{self.folder}/model_checkpoint/embeddings_entity_0.v{self.epochs}.h5'):
            print("Already trained")
            return

        config = ConfigSchema(
            entity_path=str(self.folder),

            entities={
                "entity": EntitySchema(
                    num_partitions=1,
                ),
            },

            relations=[
                RelationSchema(
                    name="edge",
                    lhs="entity",
                    rhs="entity",
                    operator="diagonal",
                )
            ],

            init_path=f"{self.folder}/init",
            dynamic_relations=True,

            dimension=self.llm.embedding_length,
            global_emb=False,
            loss_fn="logistic",
            comparator="dot",
            workers=torch.get_num_threads() // 2,
            num_epochs=self.epochs,  # Number of training epochs
            batch_size=500,  # Batch size for training
            bias=True,
            num_uniform_negs=50,
            lr=0.01,
            edge_paths=[
                f"{self.folder}/edges"
            ],

            checkpoint_path=f"{self.folder}/model_checkpoint",  # Where to store model checkpoints
        )

        convert_input_data(
            entity_configs=config.entities,
            relation_configs=config.relations,
            entity_path=config.entity_path,
            edge_paths_out=config.edge_paths,
            edge_paths_in=[self.folder / "graph_edges.csv"],
            edgelist_reader=TSVEdgelistReader(
                lhs_col=0,
                rel_col=1,
                rhs_col=2,
                delimiter=','
            ),
            dynamic_relations=config.dynamic_relations
        )

        subprocess_init = SubprocessInitializer()
        train(config, subprocess_init=subprocess_init)

        wandb.init(
            project="5_align_graph",
            group="gpt-2" if self.llm.name.startswith("gpt") else "llama-2",
            name=f"{self.llm.name}-{config.loss_fn}-{config.comparator}",

            # track hyperparameters and run metadata
            config={
                "llm": self.llm.name,
                "dataset": self.dataset_name,
                "dynamic_relations": config.dynamic_relations,
                "dimension": config.dimension,
                "global_emb": config.global_emb,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "lr": config.lr,
                "num_uniform_negs": config.num_uniform_negs,
                "loss_fn": config.loss_fn,
                "comparator": config.comparator,
                "bias": config.bias,
            }
        )

        with open(f"{self.folder}/model_checkpoint/training_stats.json", "rt") as file:
            for i, line in tqdm(enumerate(file), desc='Reporting training statistics'):
                if i % 2 != 1:
                    continue
                data = json.loads(line.rstrip('\n'))
                print(data)

                epoch = data["epoch_idx"]

                loss = data["eval_stats_chunk_avg"]["metrics"]["loss"]
                wandb.log({"train/loss": loss, "epoch": epoch})

                pos_rank = data["eval_stats_chunk_avg"]["metrics"]["pos_rank"]
                wandb.log({"train/pos_rank": pos_rank, "epoch": epoch})

                mrr = data["eval_stats_chunk_avg"]["metrics"]["mrr"]
                wandb.log({"train/mrr": mrr, "epoch": epoch})

                r1 = data["eval_stats_chunk_avg"]["metrics"]["r1"]
                wandb.log({"train/r1": r1, "epoch": epoch})

                r10 = data["eval_stats_chunk_avg"]["metrics"]["r10"]
                wandb.log({"train/r10": r10, "epoch": epoch})

                r50 = data["eval_stats_chunk_avg"]["metrics"]["r50"]
                wandb.log({"train/r50": r50, "epoch": epoch})

                auc = data["eval_stats_chunk_avg"]["metrics"]["auc"]
                wandb.log({"train/auc": auc, "epoch": epoch})

        wandb.finish()

    def build_index(self):
        self.entity_index = {}
        self.relation_index = {}

        if self._use_untrained:
            with h5py.File(f'{self.folder}/init/embeddings_entity_0.v0.h5', 'r') as hf:
                trained_embeddings = hf['embeddings'][:]
        else:
            with h5py.File(f'{self.folder}/model_checkpoint/embeddings_entity_0.v{self.epochs}.h5', 'r') as hf:
                trained_embeddings = hf['embeddings'][:]

        total_entities = 0
        with open(f'{self.folder}/entities.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for i, row in tqdm(enumerate(reader), desc='Building entity index'):
                entity_id, _ = row
                self.entity_index[entity_id] = torch.from_numpy(trained_embeddings[i, :])
                total_entities += 1

        assert trained_embeddings.shape == (
        total_entities, self.llm.embedding_length), "Trained embedding shape mismatch"

        cpu = torch.device("cpu")
        with open(f'{self.folder}/relations.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for i, row in tqdm(enumerate(reader), desc='Building relation index'):
                relation_id, relation_label = row
                self.relation_index[relation_id] = self.llm.late_embedding(relation_label).squeeze(0).squeeze(0).to(cpu)

    def node_embedding(self, entity_id: str) -> torch.Tensor:
        return self.entity_index.get(entity_id).to(self.llm._device)

    def edge_embedding(self, predicate_id: str) -> torch.Tensor:
        return self.relation_index.get(predicate_id).to(self.llm._device)

    def node_embedding_batch(self, entity_ids: List[str]) -> torch.Tensor:
        batch_embedding = torch.zeros((len(entity_ids), self.llm.embedding_length)).to(self.llm._device)
        for i, entity_id in enumerate(entity_ids):
            batch_embedding[i, :] = self.node_embedding(entity_id)
        return batch_embedding

    def edge_embedding_batch(self, predicate_ids: List[str]) -> torch.Tensor:
        batch_embedding = torch.zeros((len(predicate_ids), self.llm.embedding_length)).to(self.llm._device)
        for i, predicate_id in enumerate(predicate_ids):
            batch_embedding[i, :] = self.edge_embedding(predicate_id)
        return batch_embedding
