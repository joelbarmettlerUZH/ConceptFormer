#!/usr/bin/env python3
"""Main entry point for ConceptFormer v2 evaluation.

Usage:
    python eval_conceptformer.py --checkpoint outputs/checkpoints/best_model.pt

    # Evaluate on specific dataset:
    python eval_conceptformer.py \
        --checkpoint outputs/checkpoints/best_model.pt \
        --test-dataset TRExBite \
        --k 50
"""

from conceptformer.evaluate import main

if __name__ == "__main__":
    main()
