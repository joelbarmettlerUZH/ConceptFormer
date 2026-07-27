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
    "Gemma-3-270m": {
        "10k": [f"gemma3-270m_10k_k8_s{s}_best" for s in range(3)],
        "100k": [f"gemma3-270m_100k_k8_s{s}_best" for s in range(3)],
    },
    "Gemma-3-1b": {
        "10k": [f"gemma3-1b_10k_k8_s{s}_best" for s in range(3)],
        "100k": [f"gemma3-1b_100k_k8_s{s}_best" for s in range(3)],
    },
    "Gemma-3-4b": {
        "10k": [f"gemma3-4b_10k_k8_s{s}_best" for s in range(3)],
        "100k": [f"gemma3-4b_100k_k8_s{s}_best" for s in range(3)],
    },
}
fig2, ax = plt.subplots(figsize=(6.2, 4.2))
X = {"10k": 10_000, "100k": 100_000}
for (backbone, cells), color, marker in zip(
    GRID.items(), [BLUE, GREEN, ORANGE, BLUE, GREEN, ORANGE], "osDosD", strict=True
):
    dashed = backbone.startswith("Gemma")
    xs, ys, errs = [], [], []
    for corpus, names in cells.items():
        c = acc(names, "popqa")
        b = acc(names, "popqa", "base")
        if c and b:
            xs.append(X[corpus])
            ys.append(100 * (c[0] - b[0]))
            errs.append(100 * c[1])
    if xs:
        ax.errorbar(xs, ys, yerr=errs, fmt=f"{marker}{'--' if dashed else '-'}", color=color,
                    capsize=3, markerfacecolor="none" if dashed else color, label=backbone)
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
ABL_ANS = probe_values("faithfulness", ("ablate_answer_correct", "acc"))
ABL_OTH = probe_values("faithfulness", ("ablate_other_correct", "acc"))
# median next-token KL(base||concept) on control tasks, nats
CAP_KL = probe_values("capability", ("kl", "median"))
fig3, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(13.5, 3.8))
for data, color, marker, label in [
    (ABL_ANS, ORANGE, "o", "questioned edge removed"),
    (ABL_OTH, GREEN, "s", "unrelated edge removed"),
]:
    m = [100 * statistics.mean(data[k]) for k in KS]
    e = [100 * statistics.stdev(data[k]) for k in KS]
    ax0.errorbar(KS, m, yerr=e, fmt=f"{marker}-", color=color, capsize=3, label=label)
ax0.set_xscale("log", base=2)
ax0.set_xticks(KS, [str(k) for k in KS])
ax0.set_ylim(0, 100)
ax0.set_xlabel("concept tokens $k$")
ax0.set_ylabel("accuracy on baseline-correct probes (%)")
ax0.set_title("Edge ablation", fontsize=10)
ax0.grid(True, alpha=0.25)
ax0.legend(fontsize=8)
means = [statistics.mean(SWAP[k]) for k in KS]
errs = [statistics.stdev(SWAP[k]) for k in KS]
ax1.errorbar(KS, means, yerr=errs, fmt="o-", color=BLUE, capsize=3)
ax1.set_xscale("log", base=2)
ax1.set_xticks(KS, [str(k) for k in KS])
ax1.set_xlabel("concept tokens $k$")
ax1.set_ylabel("swap-follow rate (%)")
ax1.set_title("Counterfactual swap", fontsize=10)
ax1.grid(True, alpha=0.25)
means2 = [statistics.mean(CAP_KL[k]) for k in KS]
errs2 = [statistics.stdev(CAP_KL[k]) for k in KS]
ax2.errorbar(KS, means2, yerr=errs2, fmt="s-", color=GREEN, capsize=3)
ax2.set_xscale("log", base=2)
ax2.set_xticks(KS, [str(k) for k in KS])
ax2.set_ylim(0, 0.5)
ax2.set_xlabel("concept tokens $k$")
ax2.set_ylabel("median KL(base $\\|$ concept), nats")
ax2.set_title("Capability preservation", fontsize=10)
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
wc_x, wc_y, wc_e = [], [], []
for k in KS:
    p = TRANSFER / f"pc_k{k}_best__worldcup_1hop" / "summary.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    c, b, r = d["concept"], d["base"], d["rag"]
    wc_x.append(k)
    wc_y.append(closure(c["acc"], b["acc"], r["acc"]))
    half = (c["ci95"][1] - c["ci95"][0]) / 2
    wc_e.append(100 * half / (r["acc"] - b["acc"]))
fig4, ax = plt.subplots(figsize=(6.2, 4.0))
ax.errorbar(home_x, home_y, yerr=home_e, fmt="o-", color=BLUE, capsize=3,
            label="Wikidata (home): PopQA, unseen entities")
ax.errorbar(mq_x, mq_y, yerr=mq_e, fmt="D--", color=ORANGE, capsize=3,
            label="MetaQA (zero-shot): movies")
ax.errorbar(wc_x, wc_y, yerr=wc_e, fmt="^:", color=GREEN, capsize=3,
            label="WorldCup2014 (zero-shot): sports")
ax.set_xscale("log", base=2)
ax.set_xticks(KS, [str(k) for k in KS])
ax.set_xlabel("concept tokens $k$")
ax.set_ylabel("base$\\rightarrow$RAG gap closed (%)")
ax.set_title("Same encoder, three graphs: two foreign graphs converge\nto "
             "$\\sim$32% gap closure at $k$=32", fontsize=10)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, loc="upper left")
fig4.tight_layout()
fig4.savefig(OUT / "fig_transfer.png", dpi=150, bbox_inches="tight")
print("wrote fig_transfer.png")

# ------------------------------------------ Fig 7: cross-lingual (German) label-language effect
ML = ROOT / "data/analysis/multilingual"


def ml_cell(pattern: str) -> tuple[float, float] | None:
    """(gap-closed %, %German of distinguishable answers) for the one summary matching pattern."""
    hits = list(ML.glob(f"{pattern}/summary.json"))
    if not hits:
        return None
    d = json.loads(hits[0].read_text())
    c, b, r = d["concept"]["acc"], d["base"]["acc"], d["rag"]["acc"]
    lg = d["answer_language_of_correct_concept"]
    dist = lg["en"] + lg["localized"]
    return 100 * (c - b) / (r - b), (100 * lg["localized"] / dist if dist else 0.0)


def en_label_06b(k: int) -> tuple[float, float] | None:  # DE system, EN labels, 0.6B
    if k == 8:
        return ml_cell("pc_k8_best__popqa_abl_enlabels_desys__localized__sys-de")
    return ml_cell(f"pc_k{k}_best__popqa_desys_enlabels_0.6b_k{k}__localized__sys-de")


fig5, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.0))
# Panel A: accuracy (gap closure) vs k at 0.6B, English vs German concept labels.
xs = KS
en = [en_label_06b(k) for k in KS]
de = [ml_cell(f"pc_k{k}_best__popqa_fullde_0.6b_k{k}__localized__sys-de") for k in KS]
axa.plot(xs, [v[0] if v else None for v in en], "o-", color=BLUE, label="English-labeled concepts")
axa.plot(xs, [v[0] if v else None for v in de], "s--", color=ORANGE,
         label="German-labeled concepts")
axa.set_xscale("log", base=2)
axa.set_xticks(KS, [str(k) for k in KS])
axa.set_xlabel("concept tokens $k$")
axa.set_ylabel("base$\\rightarrow$RAG gap closed (%)")
axa.set_title("German questions, Qwen3-0.6B: accuracy follows\nthe concept label language",
              fontsize=10)
axa.grid(True, alpha=0.25)
axa.legend(fontsize=9)
# Panel B: %German answers vs model size at k8, English vs German concept labels.
sizes = ["0.6B", "1.7B", "4B"]
en_k8 = [
    en_label_06b(8),
    ml_cell("q3b17_100k_k8_s0_best__popqa_desys_enlabels_1.7b_k8__localized__sys-de"),
    ml_cell("q3b4_100k_k8_s0_best__popqa_desys_enlabels_4b_k8__localized__sys-de"),
]
de_k8 = [ml_cell("pc_k8_best__popqa_fullde_0.6b_k8__localized__sys-de"),
         ml_cell("q3b17_100k_k8_s0_best__popqa_fullde_1.7b_k8__localized__sys-de"),
         ml_cell("q3b4_100k_k8_s0_best__popqa_fullde_4b_k8__localized__sys-de")]
axb.plot(sizes, [v[1] if v else None for v in en_k8], "o-", color=BLUE,
         label="English-labeled concepts")
axb.plot(sizes, [v[1] if v else None for v in de_k8], "s--", color=ORANGE,
         label="German-labeled concepts")
axb.set_ylim(0, 100)
axb.set_xlabel("frozen backbone")
axb.set_ylabel("answers given in German (% of distinguishable)")
axb.set_title("Answer language ($k$=8): label-language effect\nwashes out as the backbone scales",
              fontsize=10)
axb.grid(True, alpha=0.25)
axb.legend(fontsize=9, loc="lower right")
fig5.tight_layout()
fig5.savefig(OUT / "fig_multilingual.png", dpi=150, bbox_inches="tight")
print("wrote fig_multilingual.png")
