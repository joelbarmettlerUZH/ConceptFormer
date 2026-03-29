"""Data loading for ConceptFormer v2.

Bridges the existing HuggingFace dataset format (TRExBite/TriREx + TRExStar graphs)
with the new model that uses integer entity/relation IDs for nn.Embedding lookup.

Key changes from v1:
- Builds entity/relation vocabularies and maps Q-IDs to integer indices.
- Pre-tokenises text segments (start, end, target) and pads per-batch via collator.
- Returns integer IDs for graph neighbours (not pre-computed float embeddings).
- Compatible with standard PyTorch DataLoader (no custom batch sampler needed).
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary: Q-ID <-> integer index
# ---------------------------------------------------------------------------

class KGVocabulary:
    """Maps Wikidata Q/P-IDs to contiguous integer indices for nn.Embedding."""

    def __init__(self):
        self.id2idx: Dict[str, int] = {}
        self.idx2id: Dict[int, str] = {}
        # Reserve index 0 for padding / unknown
        self._add("<PAD>")

    def _add(self, qid: str) -> int:
        if qid not in self.id2idx:
            idx = len(self.id2idx)
            self.id2idx[qid] = idx
            self.idx2id[idx] = qid
        return self.id2idx[qid]

    def add(self, qid: str) -> int:
        return self._add(qid)

    def __getitem__(self, qid: str) -> int:
        return self.id2idx.get(qid, 0)  # 0 = PAD/unknown

    def __len__(self) -> int:
        return len(self.id2idx)

    def __contains__(self, qid: str) -> bool:
        return qid in self.id2idx


def build_vocabularies(
    graphs: Dict[str, nx.DiGraph],
) -> Tuple[KGVocabulary, KGVocabulary]:
    """Scan all graphs to build entity and relation vocabularies."""
    entity_vocab = KGVocabulary()
    relation_vocab = KGVocabulary()

    for entity_id, G in tqdm(graphs.items(), desc="Building KG vocabularies"):
        entity_vocab.add(entity_id)
        for _, neighbour_id, edge_data in G.edges(data=True):
            entity_vocab.add(neighbour_id)
            relation_vocab.add(edge_data["id"])

    logger.info(f"Entity vocabulary: {len(entity_vocab)} entries")
    logger.info(f"Relation vocabulary: {len(relation_vocab)} entries")
    return entity_vocab, relation_vocab


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ConceptFormerDataset(Dataset):
    """Wraps a REx-style HuggingFace dataset with graph data for ConceptFormer v2.

    Each item returns:
    - Graph: integer IDs for central entity, neighbours, and relations.
    - Text: pre-tokenised start/end/target segments.
    """

    def __init__(
        self,
        rex_data,  # HuggingFace Dataset split
        graphs: Dict[str, nx.DiGraph],
        entity_vocab: KGVocabulary,
        relation_vocab: KGVocabulary,
        tokenizer: PreTrainedTokenizer,
        num_neighbors: int = 100,
        max_seq_length: int = 512,
        replace_subject: bool = False,
    ):
        self.rex_data = rex_data
        self.graphs = graphs
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.tokenizer = tokenizer
        self.num_neighbors = num_neighbors
        self.max_seq_length = max_seq_length
        self.replace_subject = replace_subject

        # Pre-filter: only keep samples whose subject is in the graph
        self.valid_indices = []
        for i in range(len(rex_data)):
            subject_id = rex_data[i]["subject"]["id"]
            if subject_id in graphs and graphs[subject_id].number_of_edges() > 0:
                self.valid_indices.append(i)

        logger.info(
            f"ConceptFormerDataset: {len(self.valid_indices)}/{len(rex_data)} samples have valid graphs"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _get_neighbors(self, G: nx.DiGraph, subject_id: str, object_id: str, predicate_id: str):
        """Extract top-N neighbours sorted by PageRank, ensuring object is included."""
        neighbour_ids, edge_ids, ranks = [], [], []
        for _, neighbour_id, edge_data in G.edges(data=True):
            neighbour_ids.append(neighbour_id)
            edge_ids.append(edge_data["id"])
            ranks.append(G.nodes[neighbour_id].get("rank", 0))

        # Sort by rank descending
        combined = sorted(zip(neighbour_ids, edge_ids, ranks), key=lambda x: x[2], reverse=True)
        neighbour_ids = [c[0] for c in combined]
        edge_ids = [c[1] for c in combined]

        # Truncate
        if len(neighbour_ids) > self.num_neighbors:
            neighbour_ids = neighbour_ids[: self.num_neighbors]
            edge_ids = edge_ids[: self.num_neighbors]

            # Ensure object is in the neighbour list
            if object_id not in neighbour_ids:
                neighbour_ids[-1] = object_id
                edge_ids[-1] = predicate_id

        # Convert to integer indices
        n_ids = [self.entity_vocab[nid] for nid in neighbour_ids]
        e_ids = [self.relation_vocab[eid] for eid in edge_ids]
        return n_ids, e_ids

    def __getitem__(self, idx: int) -> dict:
        raw_idx = self.valid_indices[idx]
        item = self.rex_data[raw_idx]

        sentence = item["sentence"]
        subject_id = item["subject"]["id"]
        subject_start, subject_end = item["subject"]["boundaries"]
        predicate_id = item["predicate"]["id"]
        object_id = item["object"]["id"]
        object_start, object_end = item["object"]["boundaries"]

        G = self.graphs[subject_id]

        # --- Graph ---
        central_entity_idx = self.entity_vocab[subject_id]
        neighbor_ids, relation_ids = self._get_neighbors(G, subject_id, object_id, predicate_id)

        # --- Text segmentation ---
        # start_text: text before subject (or up to subject end if not replacing)
        if subject_start > 0:
            if self.replace_subject:
                start_text = sentence[:subject_start].rstrip()
            else:
                start_text = sentence[:subject_end].rstrip()
        else:
            start_text = ""

        # end_text: text between subject and object
        end_text = sentence[subject_end:object_start].rstrip()

        # target_text: the object text we want to predict
        target_text = " " + sentence[object_start:object_end].strip()

        # --- Tokenise ---
        # We tokenize each segment separately. The collator will pad.
        start_ids = self.tokenizer.encode(start_text, add_special_tokens=False) if start_text else []
        end_ids = self.tokenizer.encode(end_text, add_special_tokens=False) if end_text else []

        # For target, we need to handle the tokenization carefully:
        # tokenize (end_text + target_text) and take the suffix to handle BPE boundaries.
        if end_text:
            combined = end_text + target_text
            combined_ids = self.tokenizer.encode(combined, add_special_tokens=False)
            target_ids = combined_ids[len(end_ids):]
        else:
            target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

        # If target is empty (shouldn't happen, but be safe), add a dummy
        if not target_ids:
            target_ids = [self.tokenizer.eos_token_id or 0]

        return {
            "central_entity_id": central_entity_idx,
            "neighbor_entity_ids": neighbor_ids,
            "relation_ids": relation_ids,
            "start_input_ids": start_ids,
            "end_input_ids": end_ids,
            "target_input_ids": target_ids,
            # Metadata (for evaluation)
            "subject_id": subject_id,
            "object_id": object_id,
            "predicate_id": predicate_id,
            "sentence": sentence,
        }


# ---------------------------------------------------------------------------
# Collator: dynamic padding per batch
# ---------------------------------------------------------------------------

class ConceptFormerCollator:
    """Pads variable-length fields within each batch."""

    def __init__(self, pad_token_id: int, num_neighbors: int):
        self.pad_token_id = pad_token_id
        self.num_neighbors = num_neighbors

    def __call__(self, batch: List[dict]) -> dict:
        B = len(batch)

        # --- Graph: pad neighbor lists to num_neighbors ---
        central_ids = torch.tensor([b["central_entity_id"] for b in batch], dtype=torch.long)

        max_n = max(len(b["neighbor_entity_ids"]) for b in batch)
        max_n = min(max_n, self.num_neighbors)

        neighbor_ids = torch.zeros(B, max_n, dtype=torch.long)
        relation_ids_t = torch.zeros(B, max_n, dtype=torch.long)
        neighbor_mask = torch.zeros(B, max_n, dtype=torch.bool)

        for i, b in enumerate(batch):
            n = min(len(b["neighbor_entity_ids"]), max_n)
            neighbor_ids[i, :n] = torch.tensor(b["neighbor_entity_ids"][:n], dtype=torch.long)
            relation_ids_t[i, :n] = torch.tensor(b["relation_ids"][:n], dtype=torch.long)
            neighbor_mask[i, :n] = True

        # --- Text: pad each segment ---
        start_ids, start_mask = self._pad_ids([b["start_input_ids"] for b in batch])
        end_ids, end_mask = self._pad_ids([b["end_input_ids"] for b in batch])
        target_ids, target_mask = self._pad_ids([b["target_input_ids"] for b in batch])

        return {
            "central_entity_ids": central_ids,
            "neighbor_entity_ids": neighbor_ids,
            "relation_ids": relation_ids_t,
            "neighbor_mask": neighbor_mask,
            "start_input_ids": start_ids,
            "start_attention_mask": start_mask,
            "end_input_ids": end_ids,
            "end_attention_mask": end_mask,
            "target_input_ids": target_ids,
            "target_attention_mask": target_mask,
        }

    def _pad_ids(self, id_lists: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Right-pad a list of variable-length id sequences."""
        # Ensure minimum length of 1 (some segments might be empty)
        max_len = max(max(len(ids) for ids in id_lists), 1)
        B = len(id_lists)
        padded = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros(B, max_len, dtype=torch.long)
        for i, ids in enumerate(id_lists):
            if ids:
                n = len(ids)
                padded[i, :n] = torch.tensor(ids, dtype=torch.long)
                mask[i, :n] = 1
        return padded, mask


# ---------------------------------------------------------------------------
# Factory: build datasets + loaders from existing data artifacts
# ---------------------------------------------------------------------------

def load_graphs(graph_dataset_name: str) -> Dict[str, nx.DiGraph]:
    """Load TRExStar graphs using existing dataset builders."""
    from src.Datasets.factory import trex_star_graphs_factory
    return trex_star_graphs_factory(graph_dataset_name)


def load_rex_splits(dataset_name: str):
    """Load REx sentence splits using existing dataset builders."""
    from src.Datasets.factory import rex_raw_factory
    return rex_raw_factory(dataset_name)


def create_dataloaders(
    dataset_name: str,
    graph_dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    num_neighbors: int = 100,
    max_seq_length: int = 512,
    replace_subject: bool = False,
    per_device_batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, KGVocabulary, KGVocabulary]:
    """Build train/val/test DataLoaders and KG vocabularies.

    Returns (train_loader, val_loader, test_loader, entity_vocab, relation_vocab).
    """
    logger.info(f"Loading graphs from {graph_dataset_name}...")
    graphs = load_graphs(graph_dataset_name)

    logger.info(f"Loading sentence data from {dataset_name}...")
    train_data, val_data, test_data = load_rex_splits(dataset_name)

    logger.info("Building KG vocabularies...")
    entity_vocab, relation_vocab = build_vocabularies(graphs)

    dataset_kwargs = dict(
        graphs=graphs,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        tokenizer=tokenizer,
        num_neighbors=num_neighbors,
        max_seq_length=max_seq_length,
        replace_subject=replace_subject,
    )

    train_dataset = ConceptFormerDataset(rex_data=train_data, **dataset_kwargs)
    val_dataset = ConceptFormerDataset(rex_data=val_data, **dataset_kwargs)
    test_dataset = ConceptFormerDataset(rex_data=test_data, **dataset_kwargs)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = ConceptFormerCollator(pad_token_id=pad_id, num_neighbors=num_neighbors)

    loader_kwargs = dict(
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=per_device_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, entity_vocab, relation_vocab
