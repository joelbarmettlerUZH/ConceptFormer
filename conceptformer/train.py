"""Accelerate-based training pipeline for ConceptFormer v2.

Key improvements over v1:
- HuggingFace Accelerate handles DDP, mixed precision, gradient accumulation.
- Full-sequence teacher forcing (single LLM forward pass per batch).
- Cosine LR schedule with linear warmup.
- Gradient clipping.
- Proper checkpointing via Accelerate.
"""

import logging
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from conceptformer.config import ConceptFormerConfig
from conceptformer.data import create_dataloaders
from conceptformer.model import ConceptFormerModel

logger = logging.getLogger(__name__)


def create_model_and_tokenizer(config: ConceptFormerConfig, entity_vocab_size: int, relation_vocab_size: int):
    """Load the frozen LLM backbone and create the ConceptFormer model."""
    logger.info(f"Loading LLM: {config.llm_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        config.llm_name_or_path,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32,
        attn_implementation="sdpa",  # Use PyTorch scaled-dot-product attention
    )

    config.entity_vocab_size = entity_vocab_size
    config.relation_vocab_size = relation_vocab_size

    model = ConceptFormerModel(config, llm, tokenizer)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {trainable:,} trainable / {total:,} total")

    return model, tokenizer


def train(config: ConceptFormerConfig):
    """Main training loop."""

    # --- Accelerator ---
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="wandb",
        project_dir=config.output_dir,
    )

    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    set_seed(42)

    # --- Data ---
    # Load tokenizer first (needed for dataset creation)
    tokenizer = AutoTokenizer.from_pretrained(config.llm_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader, test_loader, entity_vocab, relation_vocab = create_dataloaders(
        dataset_name=config.train_dataset_name,
        graph_dataset_name=config.graph_dataset_name,
        tokenizer=tokenizer,
        num_neighbors=config.num_neighbors,
        max_seq_length=config.max_seq_length,
        replace_subject=config.replace_subject,
        per_device_batch_size=config.per_device_batch_size,
        num_workers=config.num_workers,
    )

    # --- Model ---
    model, tokenizer = create_model_and_tokenizer(config, len(entity_vocab), len(relation_vocab))

    # --- Optimizer (only trainable params) ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # --- LR scheduler ---
    num_update_steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_training_steps = num_update_steps_per_epoch * config.num_epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_training_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy="cos",
    )

    # --- Prepare with Accelerate ---
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # --- W&B init ---
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name,
            config=vars(config),
            init_kwargs={"wandb": {"name": config.run_name or config.model_name_slug()}},
        )

    # --- Training ---
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.num_epochs}",
            disable=not accelerator.is_main_process,
        )

        for batch in progress:
            with accelerator.accumulate(model):
                outputs = model(**{k: v for k, v in batch.items() if k != "metadata"})
                loss = outputs["loss"]

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                epoch_loss += loss.item()
                num_batches += 1

                if global_step % config.log_every_n_steps == 0 and accelerator.is_main_process:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    accelerator.log(
                        {"train/loss": avg_loss, "train/lr": lr, "train/epoch": epoch + 1},
                        step=global_step,
                    )
                    progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                # Mid-epoch validation
                if config.eval_every_n_steps > 0 and global_step % config.eval_every_n_steps == 0:
                    val_loss = validate(model, val_loader, accelerator)
                    if accelerator.is_main_process:
                        accelerator.log({"val/loss": val_loss, "train/epoch": epoch + 1}, step=global_step)
                    model.train()

        # End-of-epoch validation
        val_loss = validate(model, val_loader, accelerator)
        avg_train_loss = epoch_loss / max(num_batches, 1)

        if accelerator.is_main_process:
            accelerator.log(
                {
                    "val/loss": val_loss,
                    "train/epoch_loss": avg_train_loss,
                    "train/epoch": epoch + 1,
                },
                step=global_step,
            )
            logger.info(
                f"Epoch {epoch + 1}/{config.num_epochs} — "
                f"train loss: {avg_train_loss:.4f}, val loss: {val_loss:.4f}"
            )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                save_path = checkpoint_dir / "best_model.pt"
                torch.save(
                    {
                        "graph_attention": unwrapped.graph_attention.state_dict(),
                        "kg_embeddings": unwrapped.kg_embeddings.state_dict(),
                        "config": vars(config),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "val_loss": val_loss,
                    },
                    save_path,
                )
                logger.info(f"Saved best model (val_loss={val_loss:.4f}) to {save_path}")
        else:
            patience_counter += 1
            if config.patience > 0 and patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (patience={config.patience})")
                break

    # --- Cleanup ---
    if accelerator.is_main_process:
        accelerator.end_training()
    logger.info("Training complete.")


@torch.no_grad()
def validate(model, val_loader, accelerator) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        outputs = model(**{k: v for k, v in batch.items() if k != "metadata"})
        loss = outputs["loss"]
        # Gather loss across processes
        gathered_loss = accelerator.gather(loss.unsqueeze(0)).mean().item()
        total_loss += gathered_loss
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    """Entry point for `train-conceptformer` CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ConceptFormer v2")
    parser.add_argument("--llm", default="allenai/OLMoE-1B-7B-0924", help="LLM backbone")
    parser.add_argument("--train-dataset", default="TriREx", help="Training dataset name")
    parser.add_argument("--graph-dataset", default="TRExStar", help="Graph dataset name")
    parser.add_argument("--num-pseudo-words", type=int, default=5)
    parser.add_argument("--num-neighbors", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--project", default="conceptformer-v2")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    config = ConceptFormerConfig(
        llm_name_or_path=args.llm,
        train_dataset_name=args.train_dataset,
        graph_dataset_name=args.graph_dataset,
        num_pseudo_words=args.num_pseudo_words,
        num_neighbors=args.num_neighbors,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        project_name=args.project,
        run_name=args.run_name,
        mixed_precision=args.mixed_precision,
    )

    train(config)


if __name__ == "__main__":
    main()
