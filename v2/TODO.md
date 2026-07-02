# ConceptFormer v2 — Plan / TODO (durable, survives context compaction)

Last updated: 2026-07-02. Companion to `docs/RESEARCH_FINDINGS.md` (the evidence log).

> ## ⏸️ RESUME HERE (2026-07-02 — Phase F pre-scaling analysis COMPLETE; next = the 300k scaling test)
> The full pre-scaling analysis for the compute grant is DONE and synthesized in
> **`docs/GRANT_ANALYSIS.md`** (7 sections: token-efficiency figure, RAG baseline, k-curve/knee,
> PopQA generalization, graph-faithfulness×k, capability preservation, v1 comparison).
>
> **What just landed (2026-07-02, Phase F):**
> - ✅ **Token-efficiency figure** `data/analysis/token_efficiency.png` (concept k-curve vs realistic
>   text-RAG budget curve, both axes). Concept beats RAG ~4.7x (held-out) / ~5.3x (PopQA) at 8 tokens.
>   Script `scripts/phaseF_token_efficiency_figure.py` (needs `--group viz`, matplotlib).
> - ✅ **RAG baseline** `data/analysis/rag_budget_curve.json` (`cf-rag-budget-curve`, n=300).
> - ✅ **Graph-faithfulness ACROSS k** → F13 upgraded to ✅: swap-follow scales 0.8%→34.2% (k1→k32),
>   monotone; separability holds at all k. `scripts/phaseF_graphfaith_sweep.sh`.
> - ✅ **Capability preservation across k** → F15: median KL(base||concept) tiny (0.06-0.08 nats) & ~flat as k grows;
>   concepts inert on control tasks. `scripts/phaseF_capability_sweep.sh` (+ command fixed to synthesize
>   control prompts from `generate/control.py` templates, since 100k corpus has no control tasks).
> - ✅ **v1 comparison** → F14 (`hf papers read 2504.07624`): reproduces phenomena, adds PopQA +
>   faithfulness proof. NOT a head-to-head number (different backbone/metric/task).
>
> **NEXT (the go/no-go): the 300k scaling test.** Train one k16 (or k8), 1 seed, on `cftrain_qa_300k`
> (2,412,983 rows; locked config + HPs) → does PopQA climb past 100k's k8=0.495 / k16=0.520? If yes,
> the 1M scale-up is justified and we apply for the grant.
>
> **UNCOMMITTED code (commit when user asks):** grad-accum, cache_teacher, HP knobs, best-ckpt,
> gemma-resume-cache, resumable extract-teacher-paths, `eval/counterfactual.py`,
> `cf-graph-faithfulness` / `cf-rag-budget-curve` / `cf-capability-preservation`, `log_artifact_resilient`,
> viz dep group, all `scripts/phaseF_*` + figure, `docs/GRANT_ANALYSIS.md`, F12/F13/F14 findings.
> All three gates (ruff/ty/pytest) green as of 2026-07-02.

## The situation (why this plan exists)
We discovered a **large run-to-run outcome variance** (~8 pts held-out for "identical" configs, F8),
which made fine-grained findings (F1 capacity, F2 k-curve, F7 augment) un-trustworthy (downgraded).
Phase 2 hunted for a config that makes outcomes **steerable** (low across-seed spread) WITHOUT forcing
hardware determinism. **RESOLVED:** the stabilized base is **gate-none + effective batch 32 via
grad-accum** — see LOCKED CONFIG below. Phase 3 now re-runs the downgraded findings on that base with
error bars.

Fixed backbone: Qwen3-0.6B. Corpus: `cftrain_qa_10k` / snapshot `cftrain_10k` (sha in findings doc).
Protocol: **subsample OFF (cached), report mean±std, claim only if it clears the ~1-pt floor.**

### Seed policy (decided 2026-06-20)
- **Phase 3 (all sub-phases): 3 seeds per config**, uniformly — keep measurement consistent within
  the phase (do NOT switch mid-phase even though the locked config's ~1-pt floor means big effects
  no longer strictly need 3).
- **Phase 4 and beyond: ADAPTIVE.** Screen new configs with 1–2 seeds for the shape; add a 3rd only
  on close-call contenders (<~2 pt apart); keep ≥3 seeds (error bars) on the final recipe and any
  contrast that becomes a paper claim. Spend seeds where the decision is close, not on already-
  decisive gaps. Caveat: some configs are secretly high-variance (d768/L2 hit std 4 pt on the
  stabilized base) — if 2 screening seeds disagree, add a 3rd before trusting the point.

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

- [x] **3.1 Capacity sweep** (re-run F1). DONE, group `phase31-capacity`. Curve @72k ×3:
      d512/L2 0.508±1.31 | d768/L2 0.508±4.09 | **d1024/L4 0.535±1.08 (best)** | d1536/L6 0.445±1.78.
      → **best capacity = d1024/L4 (~70M)**; 231M clearly worst (F1 re-established with error bars).
- [x] **3.2 k-curve sweep** (re-run F2) at d1024/L4. DONE, group `phase32-kcurve`. @72k ×3:
      k1 0.328 | k2 0.390 | k4 0.418 | k8 0.535 | **k16 0.637** — MONOTONE increasing, NO knee
      (old "peaks at 8" REFUTED; old undertrained sweep had k16<k8). KL falls all the way.
      **Best k DEFERRED to Phase 4** (token-cost trade). Measured fact-token cost F: median 100,
      mean 125 (k8 = 12.6× compression, k16 = 6.3×). Pushing to k=32 erodes compression → skipped.
      Missing competitor = text-RAG accuracy-vs-budget curve → **Phase 4 headline figure** (use
      REALISTIC top-PageRank retrieval, NOT answer-guaranteed, else rigged). See F2.
- [x] **3.3 Augment ablation** (re-test F7) at locked config. DONE (group `phase33-augment` +
      robustness logs). 7-prompt-mean held-out: aug-off 60.8±6.5 vs aug-on 64.6±8.6; popqa 23.0±1.7
      vs 24.3±3.6. Clean held-out-prompt-only: +5.8 pt held-out / +1.2 pt popqa. **UNDERPOWERED:**
      all point estimates trend positive (same direction as old F7) but cross-prompt seed std ~6–9 pt
      ≫ effect → not significant at n=3. NOT in the recipe on current evidence; revisit n≥6 in Phase 4
      if needed. Side-note: locked config stabilizes single-prompt (~1 pt) but NOT cross-prompt (~6 pt).
- [x] **3.4 Placement ablation** (re-run) at locked config. DONE → **F9**. prefix 0.535±1.08 |
      before 0.513±1.70 | after 0.518±0.85 | replace_entity 0.487±4.70 (worst+unstable). Placement
      within-noise among prefix/before/after; replace refuted. **RECIPE DECISION: adopt
      `before_entity`** (2.2 pt < prefix, within ~1.9× diff-SEM) for MULTI-ENTITY extensibility
      (mention-anchored, keeps surface form) — a design choice, not accuracy. Headline model must be
      trained at before_entity (placement×k interaction unmeasured). Group `phase34-placement`.
- [~] **3.5 Update RESEARCH_FINDINGS.md**: F1 re-established (peak ~70M), F2 refuted/reframed (no knee,
      token-cost view), F7 underpowered (augment trends + but not sig), F9 placement (prefix best).
      DONE for F1/F2/F7/F9 individually; remaining = a Phase-3 summary table + sync CLAUDE.md state.

## PHASE 4 — Best model + scale decision
- [ ] **4.1 Assemble the best-model recipe** (capacity + k + augment? + placement?, on locked base).
- [ ] **4.2 Decide on the 100k corpus.** Snapshot BUILT & staged (`cftrain_100k`, sha in findings);
      Gemma generation + teacher extraction NOT started (gated). Decide go/no-go once Phase 3 settles.
      If go: ~16h Gemma + ~8h teacher.

---

## MAIN-MODEL TRACK (decided 2026-06-22 — train on 100k)
Order: **generate 100k → HP+capacity sweep on 100k → train main model (k-family) → ablate.** HP sweep
precedes real training but FOLLOWS generation (sweep on the target ~2.5-epoch regime; 10k's 25-epoch
regime won't transfer). Settled & carried over: gate-none, grad-accum eff-batch-32, before_entity.
Open at scale: capacity (re-sweep — 231M overfit on 10k, may win on 10×), lr/warmup/schedule/wd/temp.

- [~] **A. Generate 100k.** Stage 1 (generate) DONE: qa.jsonl = **1,107,088 rows from 91,200 entities**
      (91% success, ~17h on Gemma/vLLM). Resume-cache (`gemma_qa:` keys, test_gemma_cache.py) worked.
      Stage 2 (tier) OOM'd FIRST attempt — `PYTORCH_CUDA_ALLOC_CONF` was set only for gen, not tier/
      extract → fragmentation. FIX: killed vLLM (gen done, freed GPU0), re-ran tier+extract with
      expandable_segments on the free GPU0 (`scripts/gen100k_tier_extract.sh`, RUNNING task bv6zb93rk,
      log /tmp/cf_tier_extract.log). tier 13.9GB stable, no OOM. Then extract → qa_distill.jsonl (~?k rows).
- [~] **B. HP sweep on 100k** (capacity NOT re-swept — "HP only, keep d1024/L4" decision). RUNNING:
      W&B sweep `nzaaivsg` (`scripts/phaseB_hp_sweep.sh`, task bnhfxm742, logs data/sweeps/hp-sweep-100k/).
      Architecture pinned (gate-none, eff-batch-32, before_entity, d1024/L4, k8, --no-cache-teacher);
      bayes over lr{5e-5,1e-4,2e-4,4e-4} x schedule{cosine,constant} x warmup-frac{0.02,0.05} x
      weight-decay{0,0.01}, 12k steps (~0.42 epoch), metric held_out/concept_acc, ~18 trials.
      NEW INFRA (gate-green): --weight-decay/--warmup-frac/--schedule (lr_factor pure helper +
      test_lr_schedule.py) and --cache-teacher (925k rows would need ~78GB cached → --no-cache-teacher
      uses live teacher). 100k train path SMOKE-validated: RAM ~10GB, trains, no OOM. 1 epoch ~25,300 steps.
- [ ] **C. Train main model** (k-family {1,2,4,8,16,32}) on 100k, before_entity, converged,
      checkpoint-select on held_out (F4 overfitting). Pick primary k.
- [ ] **D. Ablate main model**: token-efficiency figure (converged concept k-curve vs REALISTIC
      RAG-budget curve, token-cost axis), PopQA unseen-entity, prompt-robustness, full held-out.

## LOCKED TRAINING HPs (Phase B complete, 2026-06-25)
lr **1e-4**, schedule **constant**, weight-decay **0.01**, warmup-frac **0.05**. (+ architecture:
gate-none, eff-batch-32 via batch16/grad-accum2, before_entity, d1024/L4, --no-cache-teacher.)
lr-convergence check (group phaseB-lr-convergence, 40k steps): 5e-5 (0.370) ≈ 1e-4 (0.365) >> 2e-4
(0.270) — sweep's low-lr lead was the short-horizon bias; 1e-4 caught up. **Still CLIMBING at 40k
(1.6 epoch), NOT converged** (10k converged ~0.53; 100k@1.6ep already ~0.37 & rising → scale-up looks
promising). Real run needs a longer horizon (find plateau in Phase C).

## CAPABILITY PUSH — data scaling (decided 2026-06-26: scale DATA not model size; model size weakens
the efficiency story vs a 1.6B LLM). Target ~0.9 (from ~0.6). Key diagnostic (F12): token budget is
NOT the bottleneck (knee@k8) → push entities. Plan: 300k stepping-stone (confirm scaling continues)
→ 1M if still climbing. Build a data-size-vs-performance scaling curve (10k/100k/300k[/1M]).
Disk: SOLVED 2026-06-26 — freed ~1TB by deleting apertus-nano's regenerable teacher-logits cache
(data/logits 838G + data-50k/logits 163G; user OK'd, project discontinued; kept all its checkpoints/
code/records). Now **1.1TB free (69% used)** → 300k AND 1M both trivial; no further cleanup needed.

## PHASE E — graph-faithfulness proof ("learns the graph, not text compression"). Harness BUILT
(gate-green): `cf-graph-faithfulness` command + pure helpers `eval/counterfactual.py` (swap/ablate/
pool, 5 tests). Two causal interventions on a trained ckpt's INPUT graph: counterfactual edge-SWAP
(→ false neighbor; graph-faithful model follows to the FALSE answer — also disentangles from the
frozen LLM's parametric memory) + edge ABLATION (drop answer edge → that answer collapses, other
edges intact → separable). **TO RUN** on `pc_k8_best` when a GPU frees (both busy now). → becomes a
paper section + a finding. Design pre-registered in RESEARCH_FINDINGS open-Q6.

## PAUSED 2026-06-29 (GPUs handed to other work — do NOT auto-start anything; resume only on request)
Both GPUs freed. Nothing lost except k32_s1's ~4h partial (non-headline 2nd seed). RESUME steps:
- **300k corpus (extract):** ~63% was done + CACHED. Resume with ONE command (needs a GPU, ~3h):
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=N uv run --group infer python
   -m conceptformer.cli extract-teacher-paths --dataset cftrain_qa_300k --snapshot cftrain_300k
   --device cuda:0 --batch-size 32` → writes qa_distill.jsonl (resumes from teacher-path cache).
  (generate ✅ qa.jsonl, tier ✅ qa_tiered.jsonl already on disk.)
- **Seeding (Phase C2):** DONE = k4/k8/k16 ×3 seeds (on W&B). REMAINING = k32 (redo s1+s2), k2 (s1,s2),
  k1 (s1,s2). Resume = edit `scripts/phaseC2_seeds.sh` to `for k in 32 2 1` and run on a free GPU.
- **Phase D** (token-efficiency figure), **Phase E** (graph-faithfulness, harness built) — run when GPUs return.
- Current best curve already error-barred for k4/8/16 (F12); k8=0.580±0.8, k16=0.643±4.8, PopQA k8=0.495.

## Background tasks currently live (keep in sync) — overnight 2026-06-29→30, HARD STOP 8am
- GPU0: `extract-teacher-paths` 300k — RESUMABLE, **batch 64** (sweet spot: 100% GPU, 13.9GB safe,
  ~57 ex/s; batch 128 OOM'd, 32 underused). ~11.6h → ~85-90% done by 8am, resume the rest next session.
  Log /tmp/cf_300k_extract.log. For 1M, swap teacher decode to vLLM for a real (several×) speedup.
- GPU1: `pc_k32_s1` seed (100k, finishes ~5:45am, best-ckpt saved; log /tmp/cf_k32_s1.log).
- DONE THIS SESSION: Phase E graph-faithfulness (F13 — concepts carry per-edge content, NOT memory/
  blob, but holistic: swap-follow ~23%; base-only 10% vs concept 63%). Made extract-teacher-paths
  RESUMABLE (infra debt cleared, needed for 1M). Both gate-green (208 tests).
- `b63p6p5me` — GPU0: full k-family seeding (pc_k{1,2,4,8,16,32}_s{1,2}, ~3 days). Logs /tmp/cf_c2seeds.log.
- 300k snapshot DONE: `cftrain_300k` = **259,600** usable (28.8% pass on deeper tail; 300k target not
  hit but ~2.6x the 100k = fine scaling point). sha ae5aafa3...
- `b2mx0vqqx` — 300k Gemma gen + tier + extract on GPU1 (vLLM lifecycle inside). Logs
  /tmp/cf_gen300k_full.log + /tmp/cf_vllm300.log. ~2 days gen + ~1 day tier/extract. Resumable.
  PASS-RATE NOTE for 1M: 28.8% here (vs 38% @100k) → 1M needs ~3.5M+ candidates (tail degrades).
- `b8yjp1pvc` — Phase C k-family on 100k (MAIN MODEL). group phaseC-kfamily-100k, runs pc_k{1,2,4,8,16,32}.
  Locked config + HPs (lr1e-4/constant/wd0.01/warmup0.05), 100k steps (~4 ep), 1 seed, popqa-eval 200,
  checkpoint-select best held_out (model:pc_k*_best artifacts). GPU0: k1,k4,k16. GPU1: k2,k8,k32.
  Logs /tmp/cf_pc_gpu{0,1}.log. ~18h. k=8 = convergence-horizon probe. Then add 3 seeds on headline k.
  NEW INFRA (gate-green, UNCOMMITTED): best-held_out checkpoint saving (<ckpt>_best.pt + W&B artifact,
  held_out/concept_acc_best summary). Plus earlier this session: --weight-decay/--warmup-frac/--schedule,
  --cache-teacher, gemma per-entity resume cache. All tested (201 + new), ruff+ty clean.

## Phase B HP sweep RESULT (W&B sweep nzaaivsg, 18 trials @ 12k steps ~0.4 epoch)
lr dominates, LOWER better at this short horizon: mean ho_acc 5e-5=0.222 > 1e-4=0.197 > 2e-4=0.158
> 4e-4=0.142. Best trial: lr5e-5/constant/wd0.01/warmup0.05 = 0.270. constant>cosine (BUT cosine
confounded — decayed lr to 0 over the tiny horizon). wd0.01 mildly > 0. warmup 0.02~=0.05.
CAVEAT: 0.4-epoch ranking has the known low-lr bias → lr being resolved at 40k (~1.6 epoch) before
locking. SETTLED from sweep: schedule=constant (for the check), wd=0.01, warmup-frac=0.05.

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
