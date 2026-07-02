"""Phase F token-efficiency figure: concept k-curve vs realistic text-RAG budget curve.

Two panels (held_out questions, PopQA unseen entities). x = knowledge tokens paid at inference
(log scale); y = greedy exact-match accuracy. Concept curve = F12 3-seed means (mean +/- std);
RAG curve = data/analysis/rag_budget_curve.json (verbalize_budgeted, NO answer guarantee) plotted
at each budget's MEDIAN realized knowledge-token count. The story: concepts dominate the low-token
regime and reach RAG-level accuracy at ~6x fewer tokens.

Run: uv run --group viz python scripts/phaseF_token_efficiency_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAG = json.loads((ROOT / "data/analysis/rag_budget_curve.json").read_text())

# Concept k-curve, converged 100k, 3-seed mean +/- std (RESEARCH_FINDINGS F12). Token cost = k.
CONCEPT_K = [1, 2, 4, 8, 16, 32]
CONCEPT = {
    "held_out": {"mean": [0.318, 0.387, 0.472, 0.580, 0.643, 0.647],
                 "std": [0.023, 0.038, 0.028, 0.008, 0.048, 0.043]},
    "popqa": {"mean": [0.227, 0.320, 0.442, 0.495, 0.520, 0.533],
              "std": [0.010, 0.078, 0.040, 0.042, 0.004, 0.027]},
}
PANELS = [("held_out", "Held-out questions (trained entities)"),
          ("popqa", "PopQA (UNSEEN entities)")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (axis, title) in zip(axes, PANELS, strict=True):
    rag = RAG[axis]
    rag_x = [r["median_knowledge_tokens"] for r in rag if r["median_knowledge_tokens"] > 0]
    rag_y = [r["acc"] for r in rag if r["median_knowledge_tokens"] > 0]
    ax.plot(rag_x, rag_y, "s--", color="#b0592f", label="text-RAG (top-PageRank facts)", zorder=2)

    c = CONCEPT[axis]
    ax.errorbar(CONCEPT_K, c["mean"], yerr=c["std"], fmt="o-", color="#2f6fb0", capsize=3,
                label="ConceptFormer (k tokens)", zorder=3)
    for k, y in zip(CONCEPT_K, c["mean"], strict=True):
        ax.annotate(f"k={k}", (k, y), textcoords="offset points", xytext=(4, -11), fontsize=8,
                    color="#2f6fb0")

    base = next(r["acc"] for r in rag if r["budget"] == 0)
    ax.axhline(base, color="gray", ls=":", lw=1, label=f"base (no knowledge) {base:.0%}", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("knowledge tokens at inference (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("greedy exact-match accuracy")
fig.suptitle("Token efficiency: ConceptFormer vs text-RAG (frozen Qwen3-0.6B, 100k corpus)",
             fontsize=13)
fig.tight_layout()
out = ROOT / "data/analysis/token_efficiency.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
