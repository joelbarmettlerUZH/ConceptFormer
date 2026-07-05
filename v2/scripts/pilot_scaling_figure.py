"""Pilot scaling-surface figure: injection margin vs frozen-backbone size (grant teaser).

Left panel: PopQA (unseen entities) concept-over-base margin at fixed 10k-entity training data,
across backbone sizes (Qwen3-0.6B anchor + Qwen3.5-0.8B/2B), k8 and k16, text + vision ports.
Right panel: the data axis at k8 (10k -> 100k entities on Qwen3-0.6B, from the corrected F12).
Together: data scales injection UP, backbone size at fixed data scales it DOWN — the 2D
interaction is the grant's open question.

All accuracies read from data/analysis/eval_final/<ckpt>/summary.json (M7 protocol; full PopQA).

Run: uv run --group viz python scripts/pilot_scaling_figure.py
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


def margin(names: list[str]) -> tuple[float, float] | None:
    """(mean, spread) of PopQA concept-over-base margin in points, over the given reports."""
    vals = []
    for name in names:
        path = EF / name / "summary.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        vals.append(100 * (d["popqa"]["concept"]["acc"] - d["popqa"]["base"]["acc"]))
    if not vals:
        return None
    spread = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return statistics.mean(vals), spread


# x-axis: backbone size in B params (log scale).
SIZES = {"0.6B": 0.6, "0.8B": 0.8, "2B": 2.0}
SERIES = [  # label, color, marker, {size: [checkpoint names]}
    ("k8 text", "#2f6fb0", "o", {
        "0.6B": [f"p25_eff32_s{s}" for s in range(3)],
        "0.8B": [f"q35b08_k8_s{s}_best" for s in range(2)],
        "2B": [f"q35b2_k8_s{s}_best" for s in range(2)],
    }),
    ("k16 text", "#1f4f80", "s", {
        "0.8B": [f"q35b08_k16_s{s}_best" for s in range(2)],
        "2B": [f"q35b2_k16_s{s}_best" for s in range(2)],
    }),
    ("k8 vision port", "#3e8f5a", "^", {
        "0.8B": [f"q35b08_vis_k8_s{s}_best" for s in range(2)],
        "2B": [f"q35b2_vis_k8_s{s}_best" for s in range(2)],
    }),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for label, color, marker, cells in SERIES:
    xs, ys, errs = [], [], []
    for size_label, names in cells.items():
        m = margin(names)
        if m is None:
            continue
        xs.append(SIZES[size_label])
        ys.append(m[0])
        errs.append(m[1])
    ax1.errorbar(xs, ys, yerr=errs, fmt=f"{marker}-", color=color, capsize=3, label=label)
ax1.set_xscale("log")
ax1.set_xticks(list(SIZES.values()), list(SIZES.keys()))
ax1.set_xlabel("frozen backbone (log params)")
ax1.set_ylabel("PopQA margin over base (pt)")
ax1.set_title("Model axis: margin SHRINKS with size\n(fixed 10k-entity training data)")
ax1.grid(True, alpha=0.25)
ax1.legend(fontsize=8)
ax1.annotate("0.6B = Qwen3 (anchor);\n0.8B/2B = Qwen3.5", xy=(0.02, 0.03),
             xycoords="axes fraction", fontsize=7, color="gray")

# Data axis (Qwen3-0.6B, k8, corrected F12): 10k vs 100k entities, identical full-PopQA sets.
data_x = [10_000, 100_000]
data_y = [100 * (0.232 - 0.103), 100 * (0.477 - 0.103)]
ax2.plot(data_x, data_y, "o-", color="#b0592f", label="k8 text (Qwen3-0.6B)")
ax2.set_xscale("log")
ax2.set_xticks(data_x, ["10k", "100k"])
ax2.set_xlabel("training entities (log)")
ax2.set_ylabel("PopQA margin over base (pt)")
ax2.set_title("Data axis: margin GROWS with entities\n(fixed 0.6B backbone; still climbing)")
ax2.grid(True, alpha=0.25)
ax2.legend(fontsize=8)

fig.suptitle("Pilot scaling surface: the two axes pull opposite ways -> the interaction is open",
             fontsize=12)
fig.tight_layout()
out = ROOT / "data/analysis/pilot_scaling_surface.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
