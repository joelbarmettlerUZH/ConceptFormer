#!/usr/bin/env python3
"""Main entry point for ConceptFormer v2 training.

Usage:
    # Single GPU:
    python train_conceptformer.py

    # Multi-GPU (2x RTX 4090) via Accelerate:
    accelerate launch --config_file accelerate_config.yaml train_conceptformer.py

    # With custom options:
    accelerate launch --config_file accelerate_config.yaml train_conceptformer.py \
        --llm allenai/OLMoE-1B-7B-0924 \
        --train-dataset TriREx \
        --num-pseudo-words 5 \
        --batch-size 8 \
        --grad-accum 4 \
        --epochs 5
"""

from conceptformer.train import main

if __name__ == "__main__":
    main()
