"""ConceptFormer-2 paper figures (Figs 3-5): matched k-curves, data x model grid, causal probes.

All data is read live from data/analysis/ (the M7 ground truth): accuracies from eval_final/,
probe panels from probes/<ckpt>__{faithfulness,capability}/summary.json (the durable reports
written by cf-graph-faithfulness / cf-capability-preservation since 2026-07-07; the 18-cell
sweep over pc_k* regenerates them). Figures regenerate as cells land; missing cells are
skipped.

Run: uv run --group viz python scripts/cf2_figures.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EF = ROOT / "data/analysis/eval_final"
OUT = ROOT / "paper/figures"
KS = [1, 2, 4, 8, 16, 32]
BLUE, ORANGE, GREEN, DARK = "#2f6fb0", "#b0592f", "#3e8f5a", "#333333"


def acc(names: list[str], eval_set: str, field: str = "concept") -> tuple | None:
    vals = []
    for n in names:
        p = EF / n / "summary.json"
        if p.exists():
            vals.append(json.loads(p.read_text())[eval_set][field]["acc"])
    if not vals:
        return None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0), len(vals)


def curve_names(corpus: str, k: int) -> list[str]:
    if corpus == "10k":
        if k == 8:
            return [f"p25_eff32_s{s}" for s in range(3)]
        if k == 32:
            return [f"v15_k32_s{s}_best" for s in range(3)]
        return [f"p32_k{k}_s{s}" for s in range(3)]
    return [f"pc_k{k}_best" if s == 0 else f"pc_k{k}_s{s}_best" for s in range(3)]


# ---------------------------------------------------------------- Fig 3: matched k-curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
for ax, es, title in [
    (axes[0], "held_out", "Held-out questions (trained entities)"),
    (axes[1], "popqa", "PopQA (unseen entities, full benchmark)"),
]:
    for corpus, color, marker in [("10k", ORANGE, "s"), ("100k", BLUE, "o")]:
        xs, ys, errs = [], [], []
        for k in KS:
            a = acc(curve_names(corpus, k), es)
            if a:
                xs.append(k)
                ys.append(a[0])
                errs.append(a[1])
        ax.errorbar(xs, ys, yerr=errs, fmt=f"{marker}-", color=color, capsize=3,
                    label=f"{corpus} training entities")
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.set_xlabel("concept tokens $k$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(title, fontsize=10)
# Annotate the shrinking data-scaling ratio on the PopQA panel (tokens substitute for data).
for k in (8, 16, 32):
    lo, hi = acc(curve_names("10k", k), "popqa"), acc(curve_names("100k", k), "popqa")
    if lo and hi:
        axes[1].annotate(f"$\\times${hi[0] / lo[0]:.2f}", xy=(k, (lo[0] + hi[0]) / 2),
                         fontsize=8, color=DARK, ha="center")
axes[0].set_ylabel("greedy exact-match accuracy")
fig.tight_layout()
fig.savefig(OUT / "fig_kcurves.png", dpi=150, bbox_inches="tight")
print("wrote fig_kcurves.png")

# ------------------------------------------------------- Fig 4: data x model grid (margins)
GRID = {  # backbone -> corpus -> checkpoint names (k=8, before_entity)
    "Qwen3-0.6B": {
        "10k": [f"v15_be_k8_s{s}_best" for s in range(3)]
        + [f"p34_before_entity_s{s}" for s in range(3)],
        "100k": curve_names("100k", 8),
    },
    "Qwen3-1.7B": {
        "10k": [f"q3b17_10k_k8_s{s}_best" for s in range(3)],
        "100k": [f"q3b17_100k_k8_s{s}_best" for s in range(3)],
    },
    "Qwen3-4B": {
        "10k": [f"q3b4_10k_k8_s{s}_best" for s in range(3)],
        "100k": [f"q3b4_100k_k8_s{s}_best" for s in range(3)],
    },
}
fig2, ax = plt.subplots(figsize=(6.2, 4.2))
X = {"10k": 10_000, "100k": 100_000}
for (backbone, cells), color, marker in zip(
    GRID.items(), [BLUE, GREEN, ORANGE], "osD", strict=True
):
    xs, ys, errs = [], [], []
    for corpus, names in cells.items():
        c = acc(names, "popqa")
        b = acc(names, "popqa", "base")
        if c and b:
            xs.append(X[corpus])
            ys.append(100 * (c[0] - b[0]))
            errs.append(100 * c[1])
    if xs:
        ax.errorbar(xs, ys, yerr=errs, fmt=f"{marker}-", color=color, capsize=3,
                    label=backbone)
ax.set_xscale("log")
ax.set_xticks(list(X.values()), list(X.keys()))
ax.set_xlabel("training entities")
ax.set_ylabel("PopQA margin over own base (pt)")
ax.set_title("Injected-knowledge margin: data axis per frozen backbone ($k$=8)", fontsize=10)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9)
fig2.tight_layout()
fig2.savefig(OUT / "fig_grid.png", dpi=150, bbox_inches="tight")
print("wrote fig_grid.png")

# ----------------------------------------------------------- Fig 5: causal probes vs k
PROBES = ROOT / "data/analysis/probes"


def probe_values(probe: str, key: tuple[str, ...]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {}
    for k in KS:
        for name in curve_names("100k", k):
            p = PROBES / f"{name}__{probe}" / "summary.json"
            if not p.exists():
                continue
            v = json.loads(p.read_text())
            for part in key:
                v = v[part]
            out.setdefault(k, []).append(float(v))
    return out


SWAP = {k: [100 * v for v in vs]  # counterfactual swap-follow, ALL condition, %
        for k, vs in probe_values("faithfulness", ("swap_follow", "acc")).items()}
# median next-token KL(base||concept) on control tasks, nats
CAP_KL = probe_values("capability", ("kl", "median"))
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
means = [statistics.mean(SWAP[k]) for k in KS]
errs = [statistics.stdev(SWAP[k]) for k in KS]
ax1.errorbar(KS, means, yerr=errs, fmt="o-", color=BLUE, capsize=3)
ax1.set_xscale("log", base=2)
ax1.set_xticks(KS, [str(k) for k in KS])
ax1.set_xlabel("concept tokens $k$")
ax1.set_ylabel("counterfactual swap-follow (%)")
ax1.set_title("Reads the graph: rewired edge $\\rightarrow$ rewired answer,\n"
              "scaling with capacity (3 seeds)", fontsize=10)
ax1.grid(True, alpha=0.25)
means2 = [statistics.mean(CAP_KL[k]) for k in KS]
errs2 = [statistics.stdev(CAP_KL[k]) for k in KS]
ax2.errorbar(KS, means2, yerr=errs2, fmt="s-", color=GREEN, capsize=3)
ax2.set_xscale("log", base=2)
ax2.set_xticks(KS, [str(k) for k in KS])
ax2.set_ylim(0, 0.5)
ax2.set_xlabel("concept tokens $k$")
ax2.set_ylabel("median KL(base $\\|$ concept), nats")
ax2.set_title("Preserves the model: off-topic next-token\ndistribution barely moves (3 seeds)",
              fontsize=10)
ax2.grid(True, alpha=0.25)
fig3.tight_layout()
fig3.savefig(OUT / "fig_probes.png", dpi=150, bbox_inches="tight")
print("wrote fig_probes.png")

# ------------------------------------------- Fig 6: cross-graph transfer (relative gap closure)
TRANSFER = ROOT / "data/analysis/transfer"


def closure(c: float, b: float, r: float) -> float:
    return 100 * (c - b) / (r - b)


home_x, home_y, home_e = [], [], []
for k in KS:
    vals = []
    for name in curve_names("100k", k):
        p = EF / name / "summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())["popqa"]
        vals.append(closure(d["concept"]["acc"], d["base"]["acc"], d["rag"]["acc"]))
    if vals:
        home_x.append(k)
        home_y.append(statistics.mean(vals))
        home_e.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
mq_x, mq_y, mq_e = [], [], []
for k in KS:
    p = TRANSFER / f"pc_k{k}_best__metaqa_1hop_full" / "summary.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    c, b, r = d["concept"], d["base"], d["rag"]
    mq_x.append(k)
    mq_y.append(closure(c["acc"], b["acc"], r["acc"]))
    # Single checkpoint: propagate the concept Wilson CI through the closure (brackets fixed).
    half = (c["ci95"][1] - c["ci95"][0]) / 2
    mq_e.append(100 * half / (r["acc"] - b["acc"]))
fig4, ax = plt.subplots(figsize=(6.2, 4.0))
ax.errorbar(home_x, home_y, yerr=home_e, fmt="o-", color=BLUE, capsize=3,
            label="Wikidata (home): PopQA, unseen entities")
ax.errorbar(mq_x, mq_y, yerr=mq_e, fmt="D--", color=ORANGE, capsize=3,
            label="MetaQA (zero-shot): unseen graph")
ax.set_xscale("log", base=2)
ax.set_xticks(KS, [str(k) for k in KS])
ax.set_xlabel("concept tokens $k$")
ax.set_ylabel("base$\\rightarrow$RAG gap closed (%)")
ax.set_title("Same encoder, two graphs: transfer keeps the shape\nof the $k$-curve at "
             "roughly half the effect", fontsize=10)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc="upper left")
fig4.tight_layout()
fig4.savefig(OUT / "fig_transfer.png", dpi=150, bbox_inches="tight")
print("wrote fig_transfer.png")
