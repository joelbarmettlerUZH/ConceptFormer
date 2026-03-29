"""Training configuration for ConceptFormer v2."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConceptFormerConfig:
    """Complete configuration for ConceptFormer training."""

    # --- Model architecture ---
    llm_name_or_path: str = "allenai/OLMoE-1B-7B-0924"
    num_pseudo_words: int = 5
    num_neighbors: int = 100
    graph_head_layers: int = 1
    graph_head_width_multiplier: float = 1.6
    graph_head_activation: str = "gelu"
    graph_head_dropout: float = 0.1

    # --- Entity/relation embeddings ---
    # These are learned end-to-end (no BigGraph).
    # entity_vocab_size and relation_vocab_size are set from the dataset at runtime.
    entity_vocab_size: int = 0
    relation_vocab_size: int = 0
    embed_dim: int = 0  # Set from LLM at runtime

    # --- Training ---
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    per_device_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    patience: int = 3  # Early stopping patience (0 = disabled)

    # --- Data ---
    graph_dataset_name: str = "TRExStar"
    train_dataset_name: str = "TriREx"
    finetune_dataset_name: str = "TRExBite"
    max_seq_length: int = 512
    num_workers: int = 4
    replace_subject: bool = False

    # --- Precision / hardware ---
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"
    compile_model: bool = False  # torch.compile (PyTorch 2.x)

    # --- Logging ---
    project_name: str = "conceptformer-v2"
    run_name: Optional[str] = None
    log_every_n_steps: int = 50
    eval_every_n_steps: int = 500

    # --- Paths ---
    output_dir: str = "outputs"
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        self.output_dir = str(Path(self.output_dir).resolve())
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(Path(self.output_dir) / "checkpoints")

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def model_name_slug(self) -> str:
        return (
            f"cf2_{self.num_pseudo_words}pw_{self.num_neighbors}nb_"
            f"{self.graph_head_layers}x{self.graph_head_width_multiplier:.1f}"
        )
