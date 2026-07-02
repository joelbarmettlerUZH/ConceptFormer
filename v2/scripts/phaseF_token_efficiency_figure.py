"""Token-efficiency figure: concept k-curve vs text-RAG budget curves (3 retrieval modes).

Two panels (strict held-out questions, FULL PopQA unseen entities). x = knowledge tokens paid
at inference (log scale); y = greedy exact-match accuracy. All curves come from the M7-corrected
eval pipeline so both sides share eval sets:
- concept curve = `scripts/aggregate_eval_final.py --out data/analysis/kfamily_corrected.json`
  (strict held-out + full PopQA, mean +/- std over seeds);
- RAG curves = data/analysis/rag_budget_curve_{pagerank,question,summary}.json
  (`cf-rag-budget-curve --retrieval ...`): query-independent truncation, query-AWARE retrieval,
  and LLM-written budgeted summaries — so the low-budget regime is not a strawman.

Run: uv run --group viz python scripts/phaseF_token_efficiency_figure.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "data/analysis"

CONCEPT_K = [1, 2, 4, 8, 16, 32]
RAG_MODES = [  # mode -> (label, color, linestyle)
    ("pagerank", "text-RAG (top-PageRank facts)", "#b0592f", "s--"),
    ("question", "text-RAG (question-aware retrieval)", "#7a4fa3", "^--"),
    ("summary", "text summary (LLM-compressed facts)", "#3e8f5a", "v--"),
]
PANELS = [("held_out", "Held-out questions (strict, trained entities)"),
          ("popqa", "PopQA (UNSEEN entities, full benchmark)")]


def concept_curve() -> dict:
    """{axis: {"mean": [...], "std": [...]}} from the corrected k-family aggregation."""
    blob = json.loads((ANALYSIS / "kfamily_corrected.json").read_text())["table"]
    out: dict = {}
    for axis, field in (("held_out", "held_out"), ("popqa", "popqa")):
        means, stds = [], []
        for k in CONCEPT_K:
            vals = blob[str(k)][field]
            means.append(statistics.mean(vals))
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        out[axis] = {"mean": means, "std": stds}
    return out


CONCEPT = concept_curve()

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (axis, title) in zip(axes, PANELS, strict=True):
    base_acc = None
    for mode, label, color, fmt in RAG_MODES:
        path = ANALYSIS / f"rag_budget_curve_{mode}.json"
        if not path.exists():
            continue
        rag = json.loads(path.read_text())[axis]
        xs = [r["median_knowledge_tokens"] for r in rag if r["median_knowledge_tokens"] > 0]
        ys = [r["acc"] for r in rag if r["median_knowledge_tokens"] > 0]
        ax.plot(xs, ys, fmt, color=color, label=label, zorder=2)
        if base_acc is None:
            base_acc = next((r["acc"] for r in rag if r["budget"] == 0), None)

    c = CONCEPT[axis]
    ax.errorbar(CONCEPT_K, c["mean"], yerr=c["std"], fmt="o-", color="#2f6fb0", capsize=3,
                label="ConceptFormer (k tokens)", zorder=3)
    for k, y in zip(CONCEPT_K, c["mean"], strict=True):
        ax.annotate(f"k={k}", (k, y), textcoords="offset points", xytext=(4, -11), fontsize=8,
                    color="#2f6fb0")

    if base_acc is not None:
        ax.axhline(base_acc, color="gray", ls=":", lw=1,
                   label=f"base (no knowledge) {base_acc:.0%}", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("knowledge tokens at inference (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel("greedy exact-match accuracy")
fig.suptitle("Token efficiency: ConceptFormer vs text baselines (frozen Qwen3-0.6B, 100k corpus)",
             fontsize=13)
fig.tight_layout()
out = ANALYSIS / "token_efficiency.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
