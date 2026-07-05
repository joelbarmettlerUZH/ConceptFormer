"""Paired text-port vs vision-port comparison for the Qwen3.5-0.8B pilot (grant aim O4).

Pairs per-item eval-final dumps at matching (k, seed) — same frozen PopQA items, same strict
held-out split — and reports accuracy deltas with exact McNemar p-values. The question this
answers: is the VLM's vision-token interface a better landing pad for concept tokens than the
text-embedding stream?

Run:  uv run python scripts/pilot_port_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

from conceptformer.eval.stats import discordant_counts, mcnemar_exact_p

ROOT = Path("data/analysis/eval_final")
MODELS = [("q35b08", "0.8B"), ("q35b2", "2B")]  # checkpoint prefix -> display size
CELLS = [(k, s) for k in (8, 16) for s in (0, 1)]


def flags(name: str, eval_set: str) -> dict[str, bool]:
    path = ROOT / name / f"{eval_set}_items.jsonl"
    if not path.exists():
        return {}
    out: dict[str, bool] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            out[f"{row['subject_qid']}|{row['question']}"] = bool(row["concept"])
    return out


def main() -> None:
    print("# Text-port vs vision-port (paired; positive delta = vision better)\n")
    print(
        "| model | set | k | seed | text acc | vision acc | delta (pt) "
        "| b/c discordant | p (McNemar) |"
    )
    print("|---|---|--:|--:|--:|--:|--:|---|--:|")
    for prefix, size in MODELS:
        for eval_set in ("held_out", "popqa"):
            for k, s in CELLS:
                text = flags(f"{prefix}_k{k}_s{s}_best", eval_set)
                vis = flags(f"{prefix}_vis_k{k}_s{s}_best", eval_set)
                shared = sorted(set(text) & set(vis))
                if not shared:
                    continue  # cell not run (e.g. 2B vision only at k8)
                t = [text[key] for key in shared]
                v = [vis[key] for key in shared]
                t_acc, v_acc = sum(t) / len(t), sum(v) / len(v)
                b, c = discordant_counts(t, v)  # b = text-only right, c = vision-only right
                p = mcnemar_exact_p(b, c)
                print(
                    f"| {size} | {eval_set} | {k} | {s} | {t_acc:.3f} | {v_acc:.3f} | "
                    f"{100 * (v_acc - t_acc):+.1f} | {b}/{c} | {p:.2e} |"
                )
    print("\nn per row = shared items (popqa ~14k, held_out up to 2k).")


if __name__ == "__main__":
    main()
