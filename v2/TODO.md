# ConceptFormer v2 — Plan / TODO (durable, survives context compaction)

Last updated: 2026-06-19. Companion to `docs/RESEARCH_FINDINGS.md` (the evidence log).

## The situation (why this plan exists)
We discovered a **large run-to-run outcome variance** (~8 pts held-out for "identical" configs, F8),
which made fine-grained findings (F1 capacity, F2 k-curve, F7 augment) un-trustworthy (downgraded).
Phase 2 hunted for a config that makes outcomes **steerable** (low across-seed spread) WITHOUT forcing
hardware determinism. **RESOLVED:** the stabilized base is **gate-none + effective batch 32 via
grad-accum** — see LOCKED CONFIG below. Phase 3 now re-runs the downgraded findings on that base with
error bars.

Fixed backbone: Qwen3-0.6B. Corpus: `cftrain_qa_10k` / snapshot `cftrain_10k` (sha in findings doc).
Protocol: **subsample OFF (cached), report mean±std over ≥3 seeds, claim only if it clears the ~1-pt floor.**

### LOCKED CONFIG (Phase-3 base, 2026-06-19)
`--gate-mode none --batch 16 --grad-accum 2` (effective batch 32), 72k steps, k8, d1024/L4, cached
teacher, aug-off, prefix. Measured @72k ×3: held-out **0.535, std 1.08 pt, range 2.5 pt** (group
`phase25-grad-accum`, `p25_eff32_s{0,1,2}`). eff64 overshoots (worse on both axes); EMA hurt.

---

## PHASE 2 — Stabilize training (COMPLETE)
- [x] **2.1 Round 1 — noise floor + reproducibility.** baseline @36k std 1.54 pt; same-seed gap 2.5 pt
      ≈ across-seed range → sensitivity-driven, seeding doesn't collapse spread.
- [x] **2.2 Gauge sensitivity + record (F8).** Written into RESEARCH_FINDINGS.md.
- [x] **2.3 Round 2 — stabilization arms.** gate-none won (std 1.03 @36k, accuracy held); EMA worse;
      grad-clip/batch32 wider. Groups `variance-study`, `round2-stabilization`.
- [x] **2.4 Validate winner @72k.** gate-none std 2.05 (range 5 vs old ~8.5); batch32 = big accuracy
      lever (+10.5 pt) but unstable. Group `phase24-72k-validation`.
- [x] **2.5 Resolve tension + LOCK.** grad-accum batch-size curve @72k → eff32-accum wins both axes
      (0.535, range 2.5). Locked. Group `phase25-grad-accum`. (n=3; locked on user's call.)

## PHASE 3 — Re-run the invalidated experiments (locked base + ≥3 seeds, mean±std)
Each effect is only claimed if it clears the ~1-pt noise floor.

- [~] **3.1 Capacity sweep** (re-run F1). `scripts/phase31_capacity.sh` (RUNNING, task b325qhic1).
      Points spanning 23× params: d512/L2 (~10M), d768/L2 (~21M), d1536/L6 (~231M); **reuse
      `p25_eff32_*` for d1024/L4 (~70M)**. Group `phase31-capacity`, ×3 seeds. → pick best capacity.
      GPU0: d1536/L6 ×3. GPU1: d512/L2 ×3 then d768/L2 ×3. ~3.9h/run (small faster, 231M slower).
- [ ] **3.2 k-curve sweep** (re-run F2) **at the best capacity from 3.1**. → pick best k (knee).
- [ ] **3.3 Augment ablation** (re-test F7): aug-on vs aug-off, ≥3 seeds. Is +augment real?
- [ ] **3.4 Placement ablation** (re-run): prefix/before/after/replace_entity, ≥3 seeds.
- [ ] **3.5 Update RESEARCH_FINDINGS.md**: re-promote/retire F1/F2/F7 + record placement (F-new)
      with run IDs + error bars.

## PHASE 4 — Best model + scale decision
- [ ] **4.1 Assemble the best-model recipe** (capacity + k + augment? + placement?, on locked base).
- [ ] **4.2 Decide on the 100k corpus.** Snapshot BUILT & staged (`cftrain_100k`, sha in findings);
      Gemma generation + teacher extraction NOT started (gated). Decide go/no-go once Phase 3 settles.
      If go: ~16h Gemma + ~8h teacher.

---

## Background tasks currently live (keep in sync)
- `b325qhic1` — Phase 3.1 capacity sweep (running on both GPUs; logs /tmp/cf_p31_gpu{0,1}.log).

## Durable artifacts / anchors
- Evidence log: `docs/RESEARCH_FINDINGS.md` (every metric → W&B run id / artifact / repro command).
- W&B: entity `university-of-zurich`, project `conceptformer-v2`. Checkpoints = `model:<name>` artifacts.
- Branch: `conceptformer-v2`. Commit author: Joel Barmettler <joel.barmettler@uzh.ch>.

## Settled this session (do NOT redo)
- F3 (subsample = no benefit) closed. F4 (undertraining → ~40%) solid. F6 (answer-edge present) solid.
- **F8 RESOLVED → locked config** (gate-none + grad-accum2 eff32). torch RNG seeded (9a1a604).
- Gradient accumulation implemented (`--grad-accum`, `_apply_accum`, `split_microbatches`) + tested;
  effective-batch sweet spot is 32 (eff64 overshoots). Placement / grad-clip / EMA / gate_mode all
  implemented + gate-green (196 tests). **Code is committable but NOT yet committed (commit on ask).**
