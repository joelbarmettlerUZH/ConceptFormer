# ConceptFormer v2 — Plan / TODO (durable, survives context compaction)

Last updated: 2026-06-16. Companion to `docs/RESEARCH_FINDINGS.md` (the evidence log).

## The situation (why this plan exists)
We discovered a **large run-to-run outcome variance** (~8 pts held-out for "identical" configs, F8).
Root cause is under investigation (init vs CUDA vs sensitivity). Until training is **stable enough to
steer parameters**, fine-grained findings (F1 capacity, F2 k-curve, F7 augment) are **NOT trustworthy**
and are downgraded. The goal is NOT bit-reproducibility — it's **low across-run outcome spread** so
effects clear the noise band with few seeds.

Fixed backbone: Qwen3-0.6B. Corpus: `cftrain_qa_10k` / snapshot `cftrain_10k` (sha in findings doc).
Established protocol going forward: **subsample OFF (cached), report mean±std over ≥3 seeds.**

---

## PHASE 2 — Stabilize training (IN PROGRESS) — blocks everything below
Goal: drive the across-seed std down until ~3–5 pt effects are detectable with ≤3 seeds.

- [~] **2.1 Round 1 — measure the noise floor.** `scripts/variance_study.sh` (running, task bdfk4s0im).
      baseline(s0,s0repro,s1,s2) + batch64(s0,s1), aug-off/prefix/k8/d1024/L4, 36k, seeded.
      Reads: same-seed final gap (s0 vs s0repro), across-seed std, batch64 vs baseline.
      Watcher buux4huod reports the same-seed comparison when the pair converges (~2h).
- [ ] **2.2 Round 1 verdict** → record the noise floor in RESEARCH_FINDINGS.md (update F8).
- [ ] **2.3 Round 2 — stabilization arms (test several, pick winner).** Levers ready & committed
      (39c27e5): `--ema-decay`, `--gate-mode none`, `--grad-clip` (+ warmup/lr). Run each ×3 seeds,
      keep whichever MINIMIZES across-seed std while holding accuracy. Likely combine winners.
- [ ] **2.4 Lock the stabilized config** (the standard for all Phase-3 runs); document it.

## PHASE 3 — Re-run the invalidated experiments (stabilized + ≥3 seeds, mean±std)
Each effect is only claimed if it clears the Phase-2 noise band.

- [ ] **3.1 Capacity sweep** (re-run F1). d_model × n_layers, seeded. → pick best capacity.
- [ ] **3.2 k-curve sweep** (re-run F2) **at the best capacity from 3.1**. → pick best k (knee).
- [ ] **3.3 Augment ablation** (re-test F7): aug-on vs aug-off, ≥3 seeds. Is +augment real?
- [ ] **3.4 Placement ablation** (re-run): prefix/before/after/replace_entity, ≥3 seeds.
      (Preliminary single-seed checkpoints exist but are untrusted: placement_aug_prefix/after.)
- [ ] **3.5 Update RESEARCH_FINDINGS.md**: re-promote/retire F1/F2/F7 + record placement (F-new)
      with run IDs + error bars.

## PHASE 4 — Best model + scale decision
- [ ] **4.1 Assemble the best-model recipe** (capacity + k + augment? + placement? + stabilizer).
- [ ] **4.2 Decide on the 100k corpus.** Snapshot already BUILT & staged (`cftrain_100k`, sha in
      findings doc); Gemma generation + teacher extraction NOT started (gated). Decide go/no-go vs.
      "augment already broke the ceiling" once Phase 3 settles. If go: ~16h Gemma + ~8h teacher.

---

## Background tasks currently live (keep in sync)
- `bdfk4s0im` — variance study round 1 (running on both GPUs).
- `buux4huod` — patient watcher: same-seed full-trajectory comparison when seed-0 pair finishes.

## Durable artifacts / anchors
- Evidence log: `docs/RESEARCH_FINDINGS.md` (every metric → W&B run id / artifact / repro command).
- W&B: entity `university-of-zurich`, project `conceptformer-v2`. Checkpoints = `model:<name>` artifacts.
- Branch: `conceptformer-v2`. Commit author: Joel Barmettler <joel.barmettler@uzh.ch>.

## Settled this session (do NOT redo)
- F3 (subsample = no benefit, ~2× slower) — closed. F4 (undertraining → ~40%, +28pts) — solid.
  F6 (answer-edge present at budget 2048) — solid. F8 (init/variance) — observation solid, cause WIP.
- torch RNG now seeded (9a1a604). Metrics trajectories + W&B model artifacts auto-logged. Placement
  feature, grad-clip, EMA, gate_mode all implemented + gate-green (192 tests).
