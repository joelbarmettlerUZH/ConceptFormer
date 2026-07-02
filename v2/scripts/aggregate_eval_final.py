"""Aggregate eval-final reports into the corrected k-family table (mean +/- std + paired tests).

Reads data/analysis/eval_final/<checkpoint>/summary.json (+ per-item popqa dumps) produced by
`conceptformer eval-final`, groups the Phase-C k-family by k, and emits:
- per-k mean +/- std over seeds for strict held-out and FULL PopQA (word-boundary and the
  official PopQA metric), with per-seed values kept visible;
- paired exact-McNemar p-values between adjacent k's on the SHARED full-PopQA item set
  (per-seed pairing, seed-0 shown) -- the statistically honest version of "k16 > k8";
- the base/RAG brackets (checkpoint-independent).

Run:  uv run python scripts/aggregate_eval_final.py [--root data/analysis/eval_final]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from conceptformer.eval.stats import discordant_counts, mcnemar_exact_p

K_FAMILY = (1, 2, 4, 8, 16, 32)
NAME_RE = re.compile(r"^pc_k(?P<k>\d+)(?:_s(?P<seed>\d+))?_best$")


def load_reports(root: Path) -> dict[tuple[int, int], dict]:
    """{(k, seed): summary} for every Phase-C k-family report under ``root``."""
    out: dict[tuple[int, int], dict] = {}
    for summary_path in sorted(root.glob("pc_k*_best/summary.json")):
        m = NAME_RE.match(summary_path.parent.name)
        if not m:
            continue
        k, seed = int(m.group("k")), int(m.group("seed") or 0)
        out[(k, seed)] = json.loads(summary_path.read_text())
    return out


def popqa_flags(root: Path, name: str, field: str) -> dict[str, bool]:
    """{item_key: correct} from a report's per-item PopQA dump (for paired tests)."""
    flags: dict[str, bool] = {}
    items_path = root / name / "popqa_items.jsonl"
    if not items_path.exists():
        return flags
    with items_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            flags[f"{row['subject_qid']}|{row['question']}"] = bool(row[field])
    return flags


def fmt_group(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f} (n=1)"
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    per_seed = "/".join(f"{v:.3f}" for v in values)
    return f"{mean:.3f} +/- {std:.3f}  [{per_seed}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/analysis/eval_final"))
    parser.add_argument("--out", type=Path, default=None, help="also write JSON here")
    args = parser.parse_args()

    reports = load_reports(args.root)
    if not reports:
        raise SystemExit(f"no pc_k*_best/summary.json under {args.root}")

    print(f"# Corrected k-family (strict held-out + full PopQA), {len(reports)} reports\n")
    header = (
        "| k | held_out strict | popqa (word-boundary) | popqa (official) | seeds |"
    )
    print(header)
    print("|--:|---|---|---|--:|")
    table: dict[int, dict] = {}
    for k in K_FAMILY:
        seeds = sorted(s for (kk, s) in reports if kk == k)
        if not seeds:
            continue
        ho = [reports[(k, s)]["held_out"]["concept"]["acc"] for s in seeds
              if "held_out" in reports[(k, s)]]
        pq = [reports[(k, s)]["popqa"]["concept"]["acc"] for s in seeds
              if "popqa" in reports[(k, s)]]
        pq_off = [reports[(k, s)]["popqa"]["concept_official"]["acc"] for s in seeds
                  if "popqa" in reports[(k, s)]]
        table[k] = {"seeds": seeds, "held_out": ho, "popqa": pq, "popqa_official": pq_off}
        print(f"| {k} | {fmt_group(ho)} | {fmt_group(pq)} | {fmt_group(pq_off)} | {len(seeds)} |")

    any_report = next(iter(reports.values()))
    if "popqa" in any_report:
        b = any_report["popqa"]["base"]["acc"]
        r = any_report["popqa"]["rag"]["acc"]
        print(f"\nPopQA brackets (checkpoint-independent): base={b:.3f}  rag={r:.3f}")

    print("\n## Paired McNemar on full PopQA (seed-0 checkpoints, shared items)\n")
    print("| contrast | b (left-only) | c (right-only) | p (exact, 2-sided) |")
    print("|---|--:|--:|--:|")
    pairs = [(K_FAMILY[i], K_FAMILY[i + 1]) for i in range(len(K_FAMILY) - 1)]
    mcnemar: dict[str, dict] = {}
    for lo, hi in pairs:
        a = popqa_flags(args.root, f"pc_k{lo}_best", "concept")
        z = popqa_flags(args.root, f"pc_k{hi}_best", "concept")
        shared = sorted(set(a) & set(z))
        if not shared:
            continue
        b_cnt, c_cnt = discordant_counts([a[s] for s in shared], [z[s] for s in shared])
        p = mcnemar_exact_p(b_cnt, c_cnt)
        mcnemar[f"k{lo}_vs_k{hi}"] = {"b": b_cnt, "c": c_cnt, "p": p, "n_shared": len(shared)}
        print(f"| k{lo} vs k{hi} | {b_cnt} | {c_cnt} | {p:.2e} |")

    if args.out:
        args.out.write_text(json.dumps({"table": table, "mcnemar": mcnemar}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
