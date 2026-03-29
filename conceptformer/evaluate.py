"""Evaluation for ConceptFormer v2.

Computes Precision@K: for each test sample, checks whether the correct
target tokens appear within the top-K LLM predictions when concept
vectors are injected.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from conceptformer.config import ConceptFormerConfig
from conceptformer.data import (
    ConceptFormerDataset,
    KGVocabulary,
    load_graphs,
    load_rex_splits,
    build_vocabularies,
    ConceptFormerCollator,
)
from conceptformer.model import ConceptFormerModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_top_k(
    model: ConceptFormerModel,
    test_dataset: ConceptFormerDataset,
    k: int = 50,
    device: torch.device = torch.device("cuda"),
    run: Optional[wandb.sdk.wandb_run.Run] = None,
    prefix: str = "",
) -> dict:
    """Evaluate P@K on a test dataset.

    Processes samples one-at-a-time (like v1) for accurate autoregressive evaluation.
    """
    model.eval()

    hits = [0] * k
    misses = [0] * k
    total = 0
    skipped = 0

    for i in tqdm(range(len(test_dataset)), desc="Evaluating P@K"):
        item = test_dataset[i]

        # Build graph inputs
        central_id = torch.tensor([item["central_entity_id"]], device=device)
        n_ids = torch.tensor([item["neighbor_entity_ids"]], device=device)
        r_ids = torch.tensor([item["relation_ids"]], device=device)
        n_mask = torch.ones_like(n_ids, dtype=torch.bool)

        # Generate concept vectors
        pseudo_words = model.generate_concept_vectors(central_id, n_ids, r_ids, n_mask)

        # Text inputs
        start_ids = item["start_input_ids"]
        end_ids = item["end_input_ids"]
        target_ids = item["target_input_ids"]

        if not start_ids and not end_ids:
            skipped += 1
            continue

        start_t = torch.tensor([start_ids], device=device) if start_ids else torch.empty(1, 0, dtype=torch.long, device=device)
        end_t = torch.tensor([end_ids], device=device) if end_ids else torch.empty(1, 0, dtype=torch.long, device=device)
        target_t = torch.tensor([target_ids], device=device)

        # Handle empty start/end by creating minimal embedding
        if start_t.numel() == 0:
            start_t = torch.tensor([[model.tokenizer.eos_token_id or 0]], device=device)
        if end_t.numel() == 0:
            end_t = torch.tensor([[model.tokenizer.eos_token_id or 0]], device=device)

        result = model.compute_top_k_accuracy(
            pseudo_words=pseudo_words,
            start_input_ids=start_t,
            end_input_ids=end_t,
            target_input_ids=target_t,
            k=k,
        )

        target_k = result["target_k"]
        is_top_k = result["is_top_k"]

        if not is_top_k:
            for j in range(k):
                misses[j] += 1
        else:
            for j in range(target_k - 1):
                misses[j] += 1
            for j in range(target_k - 1, k):
                hits[j] += 1

        total += 1

    # Compute P@K curve
    results = {}
    for ki in range(k):
        denom = hits[ki] + misses[ki]
        results[f"P@{ki + 1}"] = hits[ki] / denom if denom > 0 else 0.0

    # Log to W&B
    if run is not None:
        report_ks = [1, 2, 3, 4, 5, 10, 15, 25, 50]
        for ki in report_ks:
            if ki <= k:
                val = results[f"P@{ki}"]
                run.log({f"{prefix}P@{ki}": val})
                run.summary[f"{prefix}P@{ki}"] = val

        # Plot P@K curve
        data = [[ki + 1, results[f"P@{ki + 1}"]] for ki in range(k)]
        table = wandb.Table(data=data, columns=["k", "P@k"])
        run.log({f"{prefix}P@k_curve": wandb.plot.line(table, "k", "P@k", title=f"{prefix}Precision@K")})

    logger.info(f"Evaluated {total} samples (skipped {skipped})")
    for ki in [1, 5, 10, 25, 50]:
        if ki <= k:
            logger.info(f"  P@{ki}: {results[f'P@{ki}']:.4f}")

    return results


def main():
    """Entry point for `eval-conceptformer` CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ConceptFormer v2")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--test-dataset", default="TRExBite", help="Test dataset name")
    parser.add_argument("--graph-dataset", default="TRExStar", help="Graph dataset name")
    parser.add_argument("--k", type=int, default=50, help="Max K for P@K evaluation")
    parser.add_argument("--project", default="conceptformer-v2-eval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_config = checkpoint["config"]

    config = ConceptFormerConfig(**{
        k: v for k, v in saved_config.items()
        if k in ConceptFormerConfig.__dataclass_fields__
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and LLM
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    # Load data
    graphs = load_graphs(args.graph_dataset)
    _, _, test_data = load_rex_splits(args.test_dataset)
    entity_vocab, relation_vocab = build_vocabularies(graphs)

    config.entity_vocab_size = len(entity_vocab)
    config.relation_vocab_size = len(relation_vocab)

    test_dataset = ConceptFormerDataset(
        rex_data=test_data,
        graphs=graphs,
        entity_vocab=entity_vocab,
        relation_vocab=relation_vocab,
        tokenizer=tokenizer,
        num_neighbors=config.num_neighbors,
        max_seq_length=config.max_seq_length,
        replace_subject=config.replace_subject,
    )

    # Build model and load weights
    model = ConceptFormerModel(config, llm, tokenizer)
    model.graph_attention.load_state_dict(checkpoint["graph_attention"])
    model.kg_embeddings.load_state_dict(checkpoint["kg_embeddings"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    # W&B
    run = wandb.init(project=args.project, config=vars(config))

    evaluate_top_k(model, test_dataset, k=args.k, device=device, run=run)

    wandb.finish()


if __name__ == "__main__":
    main()
