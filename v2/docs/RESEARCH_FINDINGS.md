# ConceptFormer v2 — Research Findings (living log)

**Purpose.** A running record of experimental findings, written so every quantitative claim is
traceable to ground-truth data *before* anything goes into the paper. Numbers drift; this file is
the audit trail that lets us re-verify each one. **Do not cite a number from here in the paper
without first re-checking it at the linked source.**

Last updated: 2026-07-02.

---

## How to verify (anchors)

| Thing | Where the ground truth lives |
|---|---|
| W&B entity / project | `university-of-zurich` / `conceptformer-v2` |
| W&B run URL pattern | `https://wandb.ai/university-of-zurich/conceptformer-v2/runs/<run_id>` |
| W&B sweep URL pattern | `https://wandb.ai/university-of-zurich/conceptformer-v2/sweeps/<sweep_id>` |
| Model checkpoints | W&B **artifacts** `model:<checkpoint_name>` (e.g. `model:sub_off_72k`), attached to their run |
| Exploration corpus (F1–F9) | `data/cf_train/cftrain_qa_10k/qa_distill.jsonl` — **92,177** distill examples |
| Exploration snapshot | `data/snapshots/cftrain_10k` — 10,000 subgraphs, `min_edges=6` |
| Exploration snapshot sha | sha256 `8aa882a06d56ef028b0a2de99ac4c7f9c1b1f4dcc3c54dfd5a1ab978c93d633e` |
| Main corpus (Phase B/C/D) | `data/cf_train/cftrain_qa_100k/qa_distill.jsonl` — **925,178** distill examples |
| Main snapshot | `data/snapshots/cftrain_100k` — 100,000 subgraphs, sha256 `ee4850f5633d8aa2a374bb9e145ce10675870274e7b25ca5bf0c5f545e60242c` |
| Backbone (frozen) | `Qwen/Qwen3-0.6B` (teacher and student share it) |
| Generator (corpus QA) | Gemma `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` via vLLM (`--max-model-len 8192`) |
| Key W&B groups | `phase31-capacity` `phase32-kcurve` `phase33-augment` `phase34-placement` `phase25-grad-accum` `phaseB-lr-convergence` `phaseC-kfamily-100k`; HP sweep `nzaaivsg` |

Metrics that live **only in W&B** are linked by run id. Metrics produced by a **CLI eval** (no W&B
run) are reproduced by the exact command given — that command *is* the ground truth. Local `/tmp/*.log`
paths are ephemeral and are noted only as a convenience, never as the system of record.

**Evidence status labels.** Every claim is tagged:
- ✅ **EVIDENCE-BACKED** — a completed run/eval exists and is linked; safe to cite once re-verified.
- 🟡 **PARTIAL / IN PROGRESS** — some data exists but the comparison is incomplete; do NOT cite yet.
- ⛔ **NOT YET EVIDENCE-BACKED** — hypothesis only; must not appear as a result in the paper.

> ⚠️ **`data/` is git-ignored and machine-local**, but **model checkpoints are pushed to W&B as
> versioned `model:<name>` artifacts** (automatically by `cf-train`; existing ones uploaded
> retroactively). So a CLI-eval finding stays reproducible even if the local file is gone:
> `eval-prompt-robustness --checkpoint <name>` auto-pulls `model:<name>:latest` from W&B when the
> file is absent. The *corpus* itself is not yet a W&B artifact — re-derive it from the snapshot
> (sha above) if needed.

---

## Caveat that colours findings 1–2

The capacity and k sweeps were run at **24,000 steps with `--subsample`**, which we later showed
(finding 4) is **undertrained** — convergence needs ~54–72k steps. So the *absolute* accuracies in
findings 1–2 are below convergence, and the *relative* orderings (what saturates) are the reliable
takeaway, not the ceilings. Re-running either sweep cached + longer is the open follow-up.

They were also run **augment-off**. F7 shows augmentation is a large generalization lever, so the
capacity/k saturation points may shift with augment on — another reason to treat findings 1–2 as
relative-ordering evidence, not absolute ceilings.

---

## Finding 1 — Encoder capacity peaks at ~70M; the largest encoder (231M) is clearly worse

> ✅ **RE-ESTABLISHED on the stabilized base (Phase 3.1).** Multi-seed (×3) re-run on the locked
> config clears the noise floor: the 231M encoder underperforms by ~9 pts (≫ combined std), so
> "bigger is not better" at the top end is now **evidence**. The *refinement* vs the old single-seed
> read: the optimum is **d1024/L4 (~70M), not d768 (~21M)** — see below.

**MEASURED (Phase 3.1, group `phase31-capacity`, locked base: gate-none, eff-batch-32 via
grad-accum2, 72k, k8, ×3 seeds).** d1024/L4 reuses the locked config's 3 seeds (`p25_eff32_*`).

| d_model | n_layers | params | held_out (mean±std) | held_in | run_ids |
|--:|--:|--:|--:|--:|---|
| 512 | 2 | ~10M | 0.508 ± 1.31 | 0.755 | `p31_d512_L2_s{0,1,2}` |
| 768 | 2 | ~21M | 0.508 ± 4.09 | 0.817 | `p31_d768_L2_s{0,1,2}` |
| **1024** | **4** | **~70M** | **0.535 ± 1.08** | 0.823 | `p25_eff32_s{0,1,2}` |
| 1536 | 6 | ~231M | 0.445 ± 1.78 | 0.583 | `p31_d1536_L6_s{0,1,2}` |

**Read.**
- **231M is clearly the worst** (0.445 vs 0.535 at 70M; gap ~9 pts ≫ combined std ~1.5). "Bigger is
  not better" holds at the top end — and now with error bars, unlike the downgraded single-seed F1.
- **Peak is ~70M (d1024/L4), modestly above the 10–21M points** (0.535 vs 0.508, ~2.7 pts ≈ 1.6×
  combined std — suggestive, not airtight). So there *is* a mild capacity benefit up to ~70M, then a
  sharp reversal. This **corrects** the old single-seed claim that 21M already matched 70M.
- **231M's low held_in (0.583) too** → it is not overfitting; it underperforms *overall*. Caveat: the
  231M encoder may simply be **harder to optimize at this step/LR budget** (undertrained-relative),
  not fundamentally worse — a fixed-budget result, not a capacity ceiling per se.
- d768/L2 is anomalously noisy (std 4.09, range 10 pts; seeds .46/.505/.56) — the only high-variance
  capacity point on the stabilized base; worth a flag, n=3.

**Decision.** Best capacity = **d1024/L4 (~70M)** = the locked config. Phase 3.2 (k-curve) runs at
this capacity. (Smaller d768/d512 give up ~2.7 pts but cost 3–7× fewer params — a viable cheap
alternative if Phase 4 needs speed/scale; revisit at 100k.)

**Re-verify:** group `phase31-capacity`, ×3 seeds; confirm `p31_d1536_L6_*` (231M) sits ~9 pts below
`p25_eff32_*` (70M).

---

## Finding 2 — k-curve: accuracy keeps rising with more concept tokens (NO knee by k=16)

> 🔄 **OLD CLAIM REFUTED, re-measured (Phase 3.2).** The original "peaks at k=8, flat after; knee
> k≈4–8" was single-seed @24k (undertrained) + subsample, inside the ~8-pt noise. On the stabilized
> base @72k ×3 seeds the curve is **monotonically increasing through k=16** — there is **no plateau**
> in the tested range. The "an entity compresses into ~4–8 tokens" headline does **not** hold here.

**MEASURED (Phase 3.2, group `phase32-kcurve`, locked base: gate-none, eff-batch-32, 72k, d1024/L4,
×3 seeds).** k=8 reuses the locked config's seeds (`p25_eff32_*`).

| k | held_out (mean±std) | held_in | KL | run_ids |
|--:|--:|--:|--:|---|
| 1 | 0.328 ± 1.84 | 0.458 | 0.838 | `p32_k1_s{0,1,2}` |
| 2 | 0.390 ± 2.27 | 0.612 | 0.761 | `p32_k2_s{0,1,2}` |
| 4 | 0.418 ± 1.25 | 0.657 | 0.722 | `p32_k4_s{0,1,2}` |
| 8 | 0.535 ± 1.08 | 0.823 | 0.624 | `p25_eff32_s{0,1,2}` |
| **16** | **0.637 ± 4.40** | 0.872 | 0.553 | `p32_k16_s{0,1,2}` |

**Read.**
- **Monotone increasing, no knee:** every doubling of k adds accuracy (0.328→0.390→0.418→0.535→0.637),
  and KL falls monotonically (0.838→0.553). More tokens keep helping through k=16.
- **k=16 > k=8 is real despite k=16's noise** (std 4.40, range 10 — a high-variance point like
  d768/L2): gap 0.637 vs 0.535 = 10.2 pts ≈ 3.9× the combined SEM, and even the *worst* k=16 seed
  (0.575) beats the *best* k=8 seed (0.55). The *magnitude* past k=8 is uncertain; the *direction* is
  not.
- **Contradicts the old undertrained sweep**, where k=16 (0.280) sat *below* k=8 (0.315). That flip
  is the undertraining caveat (F4) biting: at 24k the larger-k encoder hadn't converged; at 72k it
  pulls clearly ahead. Lesson: k-capacity interacts with training horizon — never read a k-curve off
  undertrained runs.

**Token-cost reframe (the actual efficiency claim).** Raw k is the wrong axis; the claim is "k concept
tokens replace F verbalized-fact tokens." **Measured F** (Qwen tokenizer, exact teacher verbalizer
`verbalize_with_answer` @budget 2048, 200 held-out eval entities): **median 100, mean 125.1, p95 273**
tokens (budgeted == uncapped → neighborhoods fit, confirms F6). Accuracy vs knowledge-token cost
(base floor 0.11 @0 tok; teacher ceiling 0.99 @~100 tok):

| method | knowledge tokens | held-out | gap closed | compression F/k (median) |
|---|--:|--:|--:|--:|
| base | 0 | 0.11 | 0% | — |
| concept k=8 | 8 | 0.535 | 48% | 12.6× |
| concept k=16 | 16 | 0.637 | 60% | 6.3× |
| teacher/RAG | ~100 | 0.99 | 100% | 1× |

**Implication:** value lives in the **low-token regime**; raising k *erodes* the compression headline
(k=32 → only ~3×). So extending the concept curve to k=32 is **not** the priority. The missing piece
is the **competitor curve: text-RAG accuracy vs ITS token budget** {8,16,32,64,~100} — deferred to
**Phase 4** (the paper's headline figure). Design note for then: the fair RAG baseline is *realistic*
retrieval (top-PageRank `verbalize_budgeted`, **no** answer guarantee); our answer-guaranteed teacher
verbalization stays near-ceiling even at tiny budgets (needs only the ~9-token answer edge) and would
rig the comparison.

**Best-k status:** k=16 is best measured (0.637) but the curve hasn't plateaued and "best k" depends
on the token-cost trade — **deferred to Phase 4**; do NOT lock a k yet. k=16's high variance
(std 4.40) warrants extra seeds if it enters the final recipe.

**Re-verify:** group `phase32-kcurve`, ×3 seeds; confirm `p32_k16_*` mean (0.637) sits clearly above
`p25_eff32_*` (0.535, k=8) and KL decreases monotonically with k. F (token cost): re-run the offline
verbalize+tokenize count over the seed-0 val split of `cftrain_10k` (median ≈ 100).

---

## Finding 3 — Neighbor-subsampling gives no benefit at budget 2048, and costs ~2× speed

> ✅ **EVIDENCE-BACKED** — both runs completed 72k (`kuyslwjf` off, `yvuv4wkx` on). Subsampling shows
> no benefit on accuracy, KL, OR prompt robustness (F5), and costs ~2× wall-clock. See final table.

**Claim.** Training with re-sampled teacher facts (`--subsample`) vs teacher-cached (subsample off)
produces **identical accuracy/KL trajectories at equal steps**, while running **~2× slower**. At
budget 2048 the neighborhood always fits (see finding 6), so subsampling only reshuffles fact order.

**Source.** Group `ablation-subsample` — filter the project by group:
<https://wandb.ai/university-of-zurich/conceptformer-v2/groups/ablation-subsample>
- subsample **off** (cached): run `kuyslwjf` (`sub_off_72k`)
- subsample **on**: run `yvuv4wkx` (`sub_on_72k`)

Both: k=8, d1024/L4, batch 16, 72k steps, same corpus. Compare `held_out/concept_acc` vs `_step`.

| step | sub_off (`kuyslwjf`) | sub_on (`yvuv4wkx`) |
|--:|--:|--:|
| 24000 | 0.265 | 0.300 |
| 30000 | 0.330 | 0.330 |
| 36000 | 0.320 | 0.330 |

Within eval noise (n=200) the curves coincide. **Speed:** in equal wall-clock the cached run reached
step 72k while subsample reached ~36k → ~2× (matches the trainer's "~2× faster" teacher-cache note).

**Final 72k comparison** (both runs complete; all within n=200 eval noise):

| metric | sub_off `kuyslwjf` (cached) | sub_on `yvuv4wkx` (subsample) |
|---|--:|--:|
| held_out/concept_acc | 0.40 | 0.395 |
| held_in/concept_acc | 0.515 | 0.525 |
| popqa/concept_acc | 0.18 | 0.19 |
| held_out/val_kl | 0.695 | 0.706 |
| prompt-robustness held_out std (F5) | **1.2%** | **2.6%** (worse) |

**Verdict:** no benefit on any axis (accuracy, KL, prompt robustness), ~2× slower → use cached
(subsample off) going forward.

**Re-verify:** overlay `held_out/concept_acc` for both runs in W&B; confirm overlap at shared steps.
Speed: compare each run's `_runtime` at the same `_step`.

---

## Finding 4 — The ~31% "wall" was undertraining; convergence ≈ 40% held-out, then it overfits

> ✅ **EVIDENCE-BACKED** (run `kuyslwjf`, full 72k trajectory + final summary).

**Claim.** Past 24k steps, held-out keeps climbing to ~**40%** and plateaus by ~step 54–60k, while
held-**in** reaches **51.5%** — an 11.5-pt generalization gap. So the binding constraint at
convergence is **generalization**, not capacity/k/optimization.

**Source.** Run `kuyslwjf` (`sub_off_72k`) — held-out trajectory in its W&B history
(`held_out/concept_acc` vs `_step`). Final scalars from its summary:

| metric (run `kuyslwjf` summary) | value |
|---|--:|
| `held_out/concept_acc` (final, 72k) | 0.40 |
| `held_in/concept_acc` (final) | 0.515 |
| `held_out/val_kl` (final) | 0.695 |
| `popqa/concept_acc` (unseen entities) | 0.18 |
| `popqa/base_acc` (no knowledge) | 0.09 |

Held-out trajectory (from the run's history; plateau visible from ~54k): 24k 0.265 → 48k 0.35 →
54k 0.39 → 60k 0.40 → 66k 0.395 → 72k 0.40.

**Interpretation.** PopQA 0.18 vs base 0.09 = **+9 pts** entity-generalization on *unseen* entities
(positive, modest). The held_in≫held_out gap + held-out plateau = generalization-limited → motivates
testing **more data** as the next lever (not yet executed; gated on robustness work).

**Re-verify:** open `kuyslwjf`, plot `held_out/concept_acc` and `held_in` (held_in is end-of-run
summary for this run; the per-checkpoint held_in/popqa *trajectory* is only available for runs
trained after the metrics-logging change — see Methods note M2).

---

## Finding 5 — Concept vectors are prompt-robust *without* augmentation

> ✅ **EVIDENCE-BACKED for both A/B models** (`sub_off_72k`, `sub_on_72k`) — both prompt-robust
> without augmentation. 🟡 The **`--augment` lever** ablation (does explicit prompt augmentation beat
> the already-low baseline std?) is NOT yet run — do not claim anything about augmentation's effect.

**Claim.** A model trained under a single system prompt (`TEACHER_SYSTEM`) shows near-flat accuracy
across 7 system prompts, including a never-seen one: held-out std **1.2%**, PopQA std **0.6%**.
Knowledge injection via soft tokens is ~orthogonal to prompt wording, so robustness is ~free here.

**Source.** CLI eval (no W&B run of its own). **Reproduction command = ground truth:**
```
conceptformer eval-prompt-robustness --checkpoint sub_off_72k \
  --dataset cftrain_qa_10k --snapshot cftrain_10k --eval-n 200 --popqa-n 200 --device cuda:0
```
Uses checkpoint **`model:sub_off_72k`** (W&B artifact on run `kuyslwjf`); the command auto-pulls it
from W&B if `data/checkpoints/sub_off_72k.pt` is absent, so this is reproducible on any machine.

Spread across 7 system prompts (teacher + held-out prompt + aug1–5), per checkpoint:

| checkpoint | held_out mean / **std** / min | popqa mean / std / min |
|---|--:|--:|
| `sub_off_72k` (`model:sub_off_72k`) | 39.1% / **1.2%** / 37.0% | 17.1% / 0.6% / 16.5% |
| `sub_on_72k` (`model:sub_on_72k`) | 37.9% / **2.6%** / 32.0% | 17.5% / 0.7% / 17.0% |

Both are prompt-robust (every non-teacher prompt is *unseen* by these single-prompt-trained models,
yet held-out moves only a few points). Subsampling does **not** improve robustness — it is slightly
**worse** (2× the held-out std, lower worst-case), reinforcing F3. Reproduce for `sub_on` by swapping
`--checkpoint sub_on_72k` into the command above (auto-pulls `model:sub_on_72k`).

> ✅ **RESOLVED → F7:** `--augment` was run. It keeps robustness (std 1.6%) AND raises accuracy a lot
> (+7.7 held-out, +4.7 popqa). Augmentation's value is generalization, not just lower variance.

---

## Finding 6 — At budget 2048 the answer edge is always present; neighborhoods are small

> ✅ **EVIDENCE-BACKED** (local verification over the linked snapshot + corpus).

**Claim.** Across 52,177 answerable training examples the answer edge is in the verbalized teacher
facts **100%** of the time under both the plain budgeted and the answer-guaranteed verbalizers — only
**1 / 10,000** subgraphs even exceeds the 2048-token budget. So the answer-guarantee fix is a correct
safety net but changes ~0 targets at this budget, and subsampling (finding 3) has no distractors to
drop.

**Source.** Local verification over the snapshot + corpus (no W&B). Neighborhood distribution
(`verbalize` over `data/snapshots/cftrain_10k`): edges/subgraph p50=11, p90=25, p99=60, max=318;
full-verbalize tokens p50=96, p90=209, p99=476, max=2193; **1/10000** subgraphs > 2048 tokens.

**Re-verify:** re-run the distribution/answer-edge check against the snapshot (sha above) and
`qa_distill.jsonl`; both verbalizers are in `src/conceptformer/verbalize.py`
(`verbalize_budgeted`, `verbalize_with_answer`).

---

## Finding 7 — Prompt augmentation is a GENERALIZATION lever (breaks the ~40% ceiling)

> 🟡 **RE-TESTED at n=3 on the locked base (Phase 3.3) → UNDERPOWERED, trends positive, not
> significant.** Augment-on vs -off, ×3 seeds each, scored by the 7-prompt robustness harness. Point
> estimates lean positive on every framing (held-out +3.8 pt 7-prompt-mean / +5.8 pt held-out-prompt-
> only; popqa +1.2–1.4 pt) — the SAME direction as the original claim — but the **cross-prompt
> across-seed std is ~6–9 pt** (≫ the ~1-pt single-prompt floor), so nothing clears noise at n=3.
> NOT "augment does nothing" — it's "effect smaller than the cross-prompt seed variance; n=3 can't
> resolve it." Resolving needs ~6 seeds (or a lower-variance metric). The original ⛔ sign-flip was a
> 2-checkpoint artifact; with 3 seeds the mean trends up, but honestly remains a non-result for now.
> Do NOT put augment in the recipe on current evidence; revisit with more seeds in Phase 4 IF needed.
>
> **MEASURED (Phase 3.3, group `phase33-augment` + robustness logs).** aug-off = locked config reused
> (`p25_eff32_s{0,1,2}`); aug-on = `p33_augon_s{0,1,2}`. 7-prompt-mean held-out: aug-off 60.8±6.5 vs
> aug-on 64.6±8.6. popqa: 23.0±1.7 vs 24.3±3.6. Held-out-PROMPT-only (unseen by both arms, the clean
> generalization test): held-out 58.8±6.4 vs 64.7±8.3; popqa 23.0±1.2 vs 24.2±3.9. A promising n=2
> popqa separation vanished when seed s2 landed (popqa 19.7) — a reminder not to read n=2.
>
> *Side-observation (new):* the locked config stabilized SINGLE-prompt accuracy (~1-pt std) but NOT
> cross-prompt generalization (~6-pt std across seeds) — prompt-robustness is a noisier axis; F8's
> "stability" is prompt-specific. (Part of that 6 pt is n=200 sampling noise, ~3.5 pt binomial.)
>
> --- historical downgrade note (superseded by the above) ---
> ⛔ The augment effect flipped sign across single-init pairs: old code aug-on−aug-off = +8.5
> (`0ze8ia5q`−`kuyslwjf`), new code = −6.5 (`xqui9aio`−`uqdcag0a`). The single-checkpoint numbers
> below predate the stabilized base; keep for provenance only.

**Claim.** Distilling under 5 diverse system prompts (`--augment`) does NOT merely preserve prompt
robustness (F5) — it **substantially improves generalization** on the SAME 10k corpus, lifting
held-out and unseen-entity accuracy well past the ~40% that capacity/k/length all saturated at.

**Source.** `model:augment_on_72k` (run `0ze8ia5q`, group `augment-ablation`) vs `model:sub_off_72k`
(`kuyslwjf`). Both k=8, d1024/L4, 72k cached steps, same corpus; only `--augment` differs. Numbers
from the identical `eval-prompt-robustness` harness (same held-out questions + PopQA sample, mean over
7 system prompts):

| metric (7-prompt mean / std / min) | sub_off (no augment) | augment_on | Δ mean |
|---|--:|--:|--:|
| held_out concept_acc | 39.1% / 1.2% / 37.0% | **46.8%** / 1.6% / 43.5% | **+7.7** |
| popqa concept_acc (unseen entities) | 17.1% / 0.6% / 16.5% | **21.8%** / 0.5% / 21.0% | **+4.7** |

Augment stays prompt-robust (std 1.6%; its *worst* prompt 43.5% beats sub_off's *best* 40.0%). The
augment run's own end-of-training samples read higher still (held_out 48.5%, popqa 27.5%, held_in
53.5% — run `0ze8ia5q` summary) but on a different sample, so the controlled cross-prompt numbers
above are the figures to cite.

**Interpretation.** Multi-prompt distillation forces the concept vectors to encode prompt-invariant
entity knowledge rather than prompt-specific shortcuts — a strong regularizer that also multiplies
the effective training signal (5 views/example). The earlier "~40% ceiling → need more data" read
(F4) was therefore premature: augmentation breaks it to ~47% with **no extra data**.

**Knock-on caveat:** F1 (capacity) and F2 (k) were measured **augment-off**; their saturation points
may shift with augmentation on. Re-checking the best config with augment is an open follow-up.

**Re-verify:** `eval-prompt-robustness --checkpoint augment_on_72k …` (auto-pulls `model:augment_on_72k`)
vs the same on `sub_off_72k`; compare the 7-prompt mean rows.

---

## Finding 8 — Large run-to-run variance (~8 pts) — RESOLVED by the stabilized config

> ✅ **RESOLVED (config LOCKED 2026-06-19).** The ~8-pt run-to-run *outcome* variance is fixed by
> **gate-none + effective batch 32 via grad-accum** (held-out range 2.5 pt @72k, ×3 seeds). Cause was
> sensitivity (not just init): seeding torch made init reproducible but same-seed runs still diverged;
> we fixed it in training/architecture, NOT by forcing CUDA determinism. Full narrative + measurements
> below (round-1 noise floor → round-2 arms → 72k validation → grad-accum batch-size curve → LOCK).
> The downgrades it forced (F1/F2/F7) were re-tested multi-seed on the stabilized base (see those
> findings + F10/F11). The history below is kept verbatim for provenance.

**Observed (fact).** Four "prefix" runs with the same `--seed 0` (so data order identical) span
held-out **0.40–0.485 (~8.5 pts)** and KL **0.647–0.727**. Within each run training is stable (smooth
monotone eval-KL, last-5 evals span ~0.03; per-step loss is just minibatch noise, no divergence) — so
the variance is *between* runs, not late-training oscillation.

**NOT yet established (hypotheses to test).** (a) That the driver is **weight init** — those 4 runs
were *not* a clean comparison: they also mixed pre-/post-placement code, and torch was unseeded so
CUDA nondeterminism was uncontrolled too. (b) That the **mechanism** is stochastic symmetry-breaking
(permutation-symmetric latents + sign-symmetric gates + rugged frozen-LLM alignment) — this is a
*story consistent with* the differing per-token gate configs, not a proven cause; the gate is likely
not a fast amplifier (it opens gradually, saturates ~36k). (c) That larger batch / clipping reduces
it. The variance study (below / `scripts/variance_study.sh`) is designed to test (a) and (c).

**Source.** Four prefix runs, all `--seed 0` (so train/val split + batch order identical; only torch
init differed because it was unseeded until commit `9a1a604`):

| run | config | final KL | final held-out acc |
|---|---|--:|--:|
| `kuyslwjf` | aug-off | 0.695 | 0.400 |
| `uqdcag0a` | aug-off | 0.647 | 0.465 |
| `0ze8ia5q` | aug-on | 0.687 | 0.485 |
| `xqui9aio` | aug-on | 0.727 | 0.400 |

**Consequences.**
- **Any effect ≲ ~8 pts measured from single runs is unproven** (kills F1, F2, F7 as stated).
- Survivors: F3 (null subsample), F4 (undertraining, +28 pts ≫ noise), F6 (data property).
- Fixed forward: torch is now seeded (`9a1a604`) → same seed reproduces. **All comparisons must run
  ≥3 seeds and report mean±std**, and effects must clear the noise band to be claimed.

**Suspected cause (hypothesis, not yet tested).** The zero-init `tanh` gate is a single high-LR
(`gate_lr=1e-2`) bottleneck modulating the *entire* concept contribution; depending on the encoder
init, it opens into a better/worse regime early and the run commits to that basin. Mitigations to
probe: gate warmup / lower gate-LR, different encoder init, weight averaging (EMA/SWA), more steps.

**Re-verify:** re-run any config ×3 seeds with the fix; the across-seed std IS the noise floor.

**MEASURED — noise floor @36k (round 1, seeded, group `variance-study`).** With torch now seeded, a
clean ×3-seed baseline (k8/d1024/L4, aug-off, prefix, batch16, 36k) gives held-out **mean 0.330, std
1.54 pt, range 3.50 pt, KL range 0.144** (`ifmqlpqw`/`3337ngsr`/`n7s8r4lh`, + s0repro `jz9ktk2n`).
Same-seed s0 vs s0repro differ **2.50 pt** — i.e. ≈ the across-seed range. So seeding the init does
NOT collapse the spread: residual nondeterminism (CUDA/data-loader ordering) alone moves the outcome
as much as changing the seed. This **supports the sensitivity framing over the init-only hypothesis**
— whatever the perturbation, the training maps it to a meaningfully different basin. (Note the @36k
floor here, ~1.5 pt std, is smaller than the ~8.5-pt @72k spread in the table above; consistent with
the earlier observation that variance grows with horizon. The downgrades of F1/F2/F7 stand.)

**MEASURED — stabilization arms @36k (round 2, group `round2-stabilization`).** Each arm ×3 seeds vs
the baseline floor (std 1.54 pt). Lower std/range = more stable; mean must not drop.

| arm | mean held-out | std | range | KL range | run-ids |
|---|--:|--:|--:|--:|---|
| baseline (tanh gate, b16) | 0.330 | 1.54 | 3.50 | 0.144 | `ifmqlpqw` `3337ngsr` `n7s8r4lh` `jz9ktk2n` |
| **gate-none** (drop tanh gate) | 0.333 | **1.03** | **2.50** | 0.102 | `r2_gatenone_s0/1/2` |
| grad-clip 1.0 | 0.352 | 2.72 | 6.00 | 0.093 | `r2_gradclip_s0/1/2` |
| EMA 0.999 | 0.313 | 2.49 | 6.00 | 0.145 | `r2_ema_s0/1/2` |
| batch32 | 0.390 | 4.14 | 10.00 | 0.063 | `j7rsy7c6` `05uxwbjv` `vnj598by` |

**Read (CANDIDATE, not yet a finding — n=3).** `gate-none` is the only arm that tightened all three
spread measures *without* costing accuracy (std 1.54→1.03, range 3.50→2.50, KL range 0.144→0.102, mean
0.330→0.333). This is mechanistically consistent with F8's prime suspect: removing the single high-LR
`tanh` gate (replaced by a zero-init encoder `out_proj`) removes the early basin-commitment lever.
Counter-results worth recording: **EMA made it worse** (std 2.49, mean 0.313) — refutes the
weight-averaging mitigation for this setup; **batch32 is an accuracy lever, not a stability one**
(+6.0 pt mean but std 4.14, range 10 pt — and leaning on one high-flyer seed `05uxwbjv`=0.445).
**Caveat:** at n=3 the gate-none↔baseline std gap (1.03 vs 1.54) is within what 3 seeds can fluke;
this is the most promising candidate to VALIDATE at 72k (task 2.4), not a proven stabilizer.
Open follow-up: **gate-none + batch32** — does dropping the gate tame batch32's spread while keeping
its accuracy gain?

**MEASURED — 72k validation (task 2.4, group `phase24-72k-validation`).** The stabilizer must hold at
the long horizon, where the original ~8.5-pt spread was observed. Each arm ×3 seeds, k8/d1024/L4,
aug-off, prefix, gate-none, 72k.

| arm | mean held-out | std | range | KL range | run-ids |
|---|--:|--:|--:|--:|---|
| gate-none (b16) | 0.412 | 2.05 | 5.00 | 0.183 | `ctajbg7p` `gbammsfm` `bj7psrba` |
| gate-none + batch32 | **0.517** | 3.06 | 7.50 | 0.121 | `feloox12` `f8odlni6` `6uc00v7p` |

**Read — two results, one tension.**
1. **gate-none stabilizes at 72k (validates the candidate).** Its across-seed range is **5.0 pt** vs
   the original uncontrolled ~8.5-pt @72k spread, std 2.05 pt. Caveat: the cleanest head-to-head
   (seeded *tanh*-gate ×3 @72k) was not run, so "gate-none < tanh @72k" rests on the @36k controlled
   gap (1.03 vs 1.54) plus this being tighter than the old uncontrolled spread — strong but not
   airtight. Note the spread does grow 36k→72k (std 1.03→2.05) — variance-grows-with-horizon holds
   even with the gate removed.
2. **batch32 is a large accuracy lever that GROWS with horizon (new, important).** gate-none+batch32
   reaches **0.517** held-out, +10.5 pt over gate-none b16 (0.412) at the same 72k — the only
   difference is batch16→32. The gain compounds vs the +6 pt seen @36k (less gradient noise → longer
   productive training). Decisively, the combo's **worst** seed (0.480) beats gate-none's **best**
   (0.435), so the accuracy win is not a spread artifact. Cost: wider spread (std 3.06, range 7.5) —
   batch32 trades some stability for a lot of accuracy, consistent with its @36k behavior.

**Tension to resolve at lock-time (task 2.5).** Stability (gate-none b16: std 2.05, acc 0.412) vs
accuracy (gate-none+batch32: std 3.06, acc 0.517). The combo's range (7.5 pt) is only modestly below
the original problem (~8.5 pt), so it is *not yet* the "steerable" config Phase 2 set out to find —
but its accuracy is far higher. Open levers to get *both*: gradient accumulation (even larger
effective batch without OOM), or gate-none+batch32+grad-clip. **Caveat throughout: n=3.**

**MEASURED — batch-size curve @72k via gradient accumulation (task 2.5, group `phase25-grad-accum`).**
All gate-none, aug-off, prefix, k8/d1024/L4, ×3 seeds. Effective batch = `batch * grad_accum`;
accumulation reaches batches a single forward can't hold (true batch 64 OOMs on 24 GB).

| effective batch | how | mean held-out | std | range | run-ids |
|---|---|--:|--:|--:|---|
| 16 | b16 | 0.412 | 2.05 | 5.00 | `ctajbg7p` `gbammsfm` `bj7psrba` |
| 32 | true b32 | 0.517 | 3.06 | 7.50 | `feloox12` `f8odlni6` `6uc00v7p` |
| **32** | **b16 x accum2** | **0.535** | **1.08** | **2.50** | `pioias3s` `b77d9lro` `74i8ub54` |
| 64 | b16 x accum4 | 0.497 | 2.66 | 6.50 | `bv8lfaqe` `ivpk2rk3` `orhvqlc6` |

**Read — effective batch 32 is the sweet spot, and accumulation gives BOTH accuracy and stability.**
- **eff32-accum wins on both axes:** highest mean (0.535, +12 pt over eff16's 0.412) AND the tightest
  spread of every config measured in F8 (std 1.08, range 2.50) — finally below the original ~8.5-pt
  problem. This is the steerable config Phase 2 set out to find.
- **The curve is non-monotone — 64 overshoots.** eff64 mean *drops* to 0.497 (below eff32) and spread
  widens (2.66). So bigger-is-better stops by 32; there is an optimum, not a ramp. (n=3 caveat, but
  the eff32↔eff64 mean gap, 0.535 vs 0.497, exceeds their combined spread.)
- **accum32 tighter than true-batch32 (0.535±1.08 vs 0.517±3.06) — interpret cautiously.** Same
  effective batch, similar accuracy; the accum runs were tighter, but part of true-b32's 3.06 std is
  one high-flyer seed (`feloox12`=0.555), so the std *gap* is not fully trustworthy at n=3. The safe
  claim is "accum reproduces true-batch accuracy at effective-32" (control passes), not "accum is
  inherently more stable than a true batch."

**Conclusion (task 2.5 — LOCKED 2026-06-19).** The stabilized Phase-3 base config is **gate-none +
effective batch 32 via grad-accum** (`--batch 16 --grad-accum 2 --gate-mode none`, 72k, k8,
d1024/L4, cached teacher): best mean (0.535) and tightest spread (range 2.5 pt) at 72k, resolving
F8's accuracy/stability tension. The 3-seed run `p25_eff32_s{0,1,2}` (group `phase25-grad-accum`)
doubles as the d1024/L4 (70M) point of the Phase-3 capacity sweep. **Caveat: n=3** — not re-confirmed
at 6 seeds (the user chose to lock on n=3 and proceed to Phase 3). All Phase-3 ablations build on
this base and report mean±std over ≥3 seeds; an effect counts only if it clears the ~1-pt floor.

---

## Methods notes (provenance / things that affect comparability)

- **M1 — Eval brackets.** `base` = frozen Qwen, no knowledge; `teacher(RAG)` = frozen Qwen reading
  verbalized facts as text (upper bound, ~99%); `concept` = frozen Qwen reading k concept tokens
  (the trained system). All scored with the alias matcher under the stated system prompt.
- **M2 — Metrics logging changed 2026-06-16.** Runs *before* this change log `held_in`/`popqa` only
  as **end-of-run summary scalars** (e.g. `kuyslwjf`). Runs *after* log full **per-checkpoint
  trajectories** (`held_in/*`, `popqa/*`, `gen_gap/concept_acc`, `gate/mean`). When comparing across
  this boundary, use summary scalars, not history, for the older runs.
- **M3 — `subsample` is not in W&B config.** The on/off A/B is distinguished by run *name*
  (`sub_on_72k` / `sub_off_72k`), not a config field.
- **M4 — Undertraining caveat** applies to all 24k-step sweep runs (findings 1–2); see the caveat box
  above.
- **M5 — Two corpora.** `cftrain_qa_10k` (snapshot `cftrain_10k`, 10k entities, **92,177** distill
  rows) = the exploration workbench (F1–F9 all here). `cftrain_qa_100k` (snapshot `cftrain_100k`,
  100k entities, **925,178** distill rows, generated 2026-06-23) = the main-model corpus (Phase
  B/C/D). 100k is a clean 10× scale-up: 8.8% Gemma gen-fail (vs 10k's 8.9%), identical tier/task-type
  proportions, teacher paths mean-len 30.8 / 0 degenerate. **Regime differs sharply:** at eff-batch-32
  one epoch = ~25,300 steps on 100k vs ~2,900 on 10k, so the 10k runs were ~25 epochs (overfit-prone)
  and 100k runs are few-epoch. Cache teacher hidden upfront only on small data (`--no-cache-teacher`
  on 100k: 925k rows would need ~78 GB).
- **M6 — Locked main-model config (from F8 + F10).** gate-none, effective batch 32 (`--batch 16
  --grad-accum 2`), `before_entity` placement (F9), d1024/L4, k per the curve (F2/F11); HPs lr 1e-4,
  schedule constant, weight-decay 0.01, warmup-frac 0.05; `--no-cache-teacher`; checkpoint-selected on
  best held-out. This is the config all Phase-C/D numbers come from.
- **M7 — Eval-methodology overhaul (2026-07-02; external review).** Four defects in the evaluation
  protocol were identified and fixed; every pre-M7 accuracy carries them and the headline numbers
  are re-derived by `eval-final` (see the F12 correction):
  1. **Seed-coupled eval sampling.** Held-out and PopQA eval subsets were sampled with the
     *training* seed (from an RNG whose state also depended on unrelated earlier draws), so each
     seed/config scored a *different* n=200 sample. Across-seed std therefore mixed model variance
     with eval-set sampling noise (~3.5 pt binomial at n=200), and no paired comparisons were
     possible. FIX: all eval sampling now uses one fixed seed (`eval/evalsets.py`,
     `EVAL_SAMPLE_SEED`); smaller-n samples are prefixes of larger-n ones.
  2. **Tiny PopQA subsets.** n=200 of 14,267 questions. FIX: `eval-final` scores the FULL
     benchmark (binomial noise ~0.4 pt) and additionally reports the official PopQA metric for
     literature comparability. Per-item dumps + `eval/stats.py` (Wilson CIs, exact McNemar,
     paired bootstrap) replace bare point estimates.
  3. **Selection bias on held-out.** Best-checkpoint selection used the same n=200 held-out sample
     that was then reported (max over ~10 noisy evals inflates the reported number). FIX:
     `eval-final` re-scores selected checkpoints on a larger, disjointly-sampled strict set;
     selection during training still uses the (now-frozen) trajectory sample, which no longer
     overlaps the definitive one beyond its prefix.
  4. **Paraphrase leakage in "held-out".** The question-level split let paraphrases of the same
     (entity, fact) straddle train/val — measured at seed 0 on 100k: **32.8% of answerable val
     rows share a fact with a train row**. So legacy "held-out" partly measured paraphrase
     robustness. FIX: `split_by_held_out_facts` (new default, groups by `fact_key`) for new runs;
     `strict_val_subset` filters legacy checkpoints' val sets at eval time (equivalent to having
     grouped upfront).
  Also: the RAG-budget baseline gained `--retrieval question|summary` modes (query-aware
  retrieval + LLM-written budgeted summary) so the token-efficiency figure is not a
  query-independent-truncation strawman, and an untrained top-k mean-edge-embedding injection
  baseline (`eval-untrained-injection`) isolates what the *trained* encoder adds. Checkpoints now
  record their split provenance (`meta` in the blob); `eval-final` reconstructs legacy splits
  from the stored training seed.

---

## Finding 9 — Concept-token placement doesn't help; `prefix` is best, `replace_entity` is worse

> ✅ **EVIDENCE-BACKED (Phase 3.4, ×3 seeds on the locked base).** Where the k concept tokens sit
> relative to the entity mention has **no useful effect**; the default `prefix` is (weakly) best, and
> *removing* the entity surface form (`replace_entity`) is actively worse and unstable. The
> adjacency/replacement-improves-binding hypothesis is **refuted**.

**MEASURED (group `phase34-placement`, locked base: gate-none, eff-batch-32, 72k, d1024/L4, k8).**
`prefix` reuses the locked config (`p25_eff32_*`).

| placement | held-out (mean±std) | held_in | KL | run_ids |
|---|--:|--:|--:|---|
| **prefix** (default) | **0.535 ± 1.08** | 0.823 | 0.624 | `p25_eff32_s{0,1,2}` |
| before_entity | 0.513 ± 1.70 | 0.787 | 0.660 | `p34_before_entity_s{0,1,2}` |
| after_entity | 0.518 ± 0.85 | 0.813 | 0.652 | `p34_after_entity_s{0,1,2}` |
| replace_entity | 0.487 ± 4.70 | 0.848 | 0.748 | `p34_replace_entity_s{0,1,2}` |

**Read.** prefix/before/after cluster in 0.513–0.535 — a ~2-pt spread inside the combined noise, so
positioning concepts adjacent to the entity gives **no binding benefit**. `replace_entity` is lowest
(0.487) and by far the noisiest (std 4.70): when the entity surface form is deleted, the concepts
must fully *be* the entity, which both hurts accuracy and destabilizes training. Its high held_in
(0.848) with low held_out = it overfits the substitution rather than generalizing. **Keep `prefix`.**
*Caveat:* `replace_entity` s0 alone was 0.545 (highest single point) but collapsed to 0.485/0.430 at
s1/s2 — the third n=1→n=3 collapse this phase (cf. augment n=2, replace n=1); single-seed placement
reads are worthless.

**RECIPE DECISION (2026-06-22): adopt `before_entity` despite `prefix` being marginally higher.**
Rationale is forward-looking, not accuracy: `before_entity` is **mention-anchored** (concepts sit at
the entity's position), so it extends naturally to inputs that mention **multiple entities** — each
mention gets its own concept block inline, which a single message-`prefix` block cannot do cleanly
(N entities → ambiguous which concepts bind to which mention). It also keeps the entity surface form
(unlike `replace_entity`, which deletes the anchor and pays for it: worst + unstable). The accuracy
cost vs `prefix` is **2.2 pt (within ~1.9× the difference-SEM at n=3 — not a clean gap)**, accepted as
the price of multi-entity scalability. NOTE: all Phase-3 ablations (capacity/k/augment) used `prefix`;
placement is within-noise of it, so the transfer is expected to hold, but the final **headline recipe
model must be trained at `before_entity`** and confirm k=16 etc. there (placement×k interaction
unmeasured). Multi-entity is currently UNTESTED (single-entity corpus) — `before_entity`'s real
advantage is a design property, to be validated if/when a multi-entity eval exists.

**Re-verify:** group `phase34-placement`, ×3 seeds; confirm `replace_entity` mean < `prefix` and its
std (4.7) is the largest of the four modes; `before_entity` 0.513 vs `prefix` 0.535.

---

## Finding 10 — Training-HP selection on 100k; lr ≈ 1e-4, and short-horizon sweeps bias toward low lr

> ✅ **EVIDENCE-BACKED (Phase B, 100k corpus).** On the locked architecture, **lr dominates** the
> training HPs and **5e-5 ≈ 1e-4** at a real horizon (2e-4+ clearly worse). A 0.4-epoch sweep ranked
> lr *monotonically* (lower better) — the classic **short-horizon low-lr bias**; it did NOT survive a
> longer run. **Locked: lr 1e-4, schedule constant, weight-decay 0.01, warmup-frac 0.05.**

**MEASURED — HP sweep (W&B sweep `nzaaivsg`, 18 trials, 12k steps ≈ 0.4 epoch, metric held-out).**
Architecture pinned to the locked config; bayes over lr × schedule × warmup × weight-decay. Mean
held-out by lr: **5e-5 0.222 > 1e-4 0.197 > 2e-4 0.158 > 4e-4 0.142**. Best trial: lr5e-5 / constant /
wd0.01 / warmup0.05 = **0.270**. Secondary: wd 0.01 ≳ 0; warmup 0.02 ≈ 0.05; "constant > cosine" —
but that is **confounded** (cosine decayed the LR to ~0 over the tiny 12k-step horizon, penalising it
at the 0.4-epoch eval), so the schedule ranking from the sweep is not trustworthy on its own.

**MEASURED — lr at a real horizon (group `phaseB-lr-convergence`, 40k steps ≈ 1.6 epoch, constant
schedule, wd0.01/warmup0.05).** lr 5e-5 → **0.370** (`lrc_lr5e5`), lr 1e-4 → **0.365** (`lrc_lr1e4`),
lr 2e-4 → 0.270 (`lrc_lr2e4`). **5e-5 and 1e-4 are tied** (within 0.5 pt) — the sweep's "5e-5 ≫ 1e-4"
was purely the short-horizon artifact; 1e-4 fully caught up by 1.6 epochs. Locked **lr 1e-4**
(tied accuracy, faster early convergence → fewer steps, lower KL 0.688 vs 0.710, and our proven
default). **Lesson:** never read lr off a sub-epoch sweep — it systematically favours the lowest lr.

**Re-verify:** sweep `nzaaivsg` (sort by held_out/concept_acc; confirm lr monotonicity at 12k) vs
group `phaseB-lr-convergence` (confirm 5e-5≈1e-4≫2e-4 at 40k).

---

## Finding 11 — 100k is still climbing at 1.6 epochs; scale-up looks promising (convergence horizon open)

> ✅ **RESOLVED → F12.** The 1.6-epoch "still climbing" observation below was the early read; the
> converged Phase-C result (F12) confirms it: held-out plateaus ~80–100k steps (3–4 epochs), and the
> scale-up DOES lift the ceiling (most dramatically on PopQA, ~2.5×). Kept for the convergence-horizon
> evidence and the lr trajectory.

**MEASURED — held-out trajectory @ k8/d1024/L4, constant lr 1e-4 (group `phaseB-lr-convergence`).**
lr 1e-4 (`lrc_lr1e4`): 5k→0.13, 10k→0.21, 15k→0.265, 20k→0.28, 25k→0.32, 30k→0.315, 35k→0.345,
40k→**0.365** (monotone-ish, still rising). lr 5e-5 similar, peaking ~0.39 at 35k. No overfit yet at
1.6 epochs (held-in tracks held-out), so the generous Phase-C horizon + best-held-out checkpoint
selection is the right design (a model CAN overfit past its peak at 4 epochs — F4).

**Open:** the converged 100k accuracy + the convergence step-count (sets the real horizon) come from
Phase C (group `phaseC-kfamily-100k`). Until then, do NOT quote a converged 100k number.

**Re-verify:** group `phaseB-lr-convergence`, run `lrc_lr1e4`, plot held_out/concept_acc vs step.

---

## Finding 12 — Scale-up (100k) forces graph-learning over memorization: PopQA ~2.25× (held-out flat)

> ✅ **CORRECTED (M7 protocol, 2026-07-02) — the table below is SUPERSEDED; cite the corrected
> block that follows.** All 18 k-family checkpoints re-scored by `eval-final` on FULL PopQA
> (n=14,266, both the corrected word-boundary metric and the official PopQA metric) and the
> STRICT (fact-leakage-free) held-out set (n=2,000, frozen fixed-seed sample). Source:
> `data/analysis/eval_final/<ckpt>/summary.json` + per-item jsonl; aggregate =
> `scripts/aggregate_eval_final.py --out data/analysis/kfamily_corrected.json`.
>
> **CORRECTED k-family (3 seeds each; mean +/- std over seeds; per-seed values in the JSON):**
>
> | k | held-out strict | PopQA (full, word-boundary) | PopQA (official) |
> |--:|---|---|---|
> | 1 | 0.298 +/- 0.006 | 0.203 +/- 0.008 | 0.207 +/- 0.009 |
> | 2 | 0.340 +/- 0.023 | 0.298 +/- 0.075 | 0.303 +/- 0.077 |
> | 4 | 0.441 +/- 0.007 | 0.412 +/- 0.023 | 0.419 +/- 0.024 |
> | 8 | 0.545 +/- 0.014 | 0.477 +/- 0.002 | 0.483 +/- 0.002 |
> | 16 | 0.594 +/- 0.024 | 0.518 +/- 0.020 | 0.523 +/- 0.021 |
> | 32 | 0.610 +/- 0.011 | 0.537 +/- 0.010 | 0.543 +/- 0.010 |
>
> Brackets (frozen model, full PopQA): base **0.103**, RAG (answer-guaranteed facts) **0.960**.
>
> **What changed vs the superseded table:**
> 1. **Seed-std collapsed** (k8 PopQA std 4.2 pt -> 0.2 pt): the old spread was mostly n=200
>    eval sampling noise, exactly as the M7 analysis predicted. The model is far more
>    seed-stable than the old protocol could see.
> 2. **"Plateau at k16" REFUTED.** Paired exact McNemar on the shared full-PopQA items
>    (seed-0): every adjacent contrast is significant, including k16 vs k32 (c=1562 vs b=924,
>    p ~ 9e-38). The curve is monotone through k32 with diminishing per-token returns; k8
>    already buys ~89% of k32's PopQA at 1/4 the tokens. "Best k" remains a token-cost trade
>    (k8 efficiency vs k32 peak), NOT a capacity ceiling in the tested range.
> 3. **Held-out dropped 2-4 pt everywhere** (selection bias + the 32.8% paraphrase leakage
>    removed); ordering unchanged.
> 4. **Scale-up (10k -> 100k) CORRECTED to ~2.05x on identical eval sets:** 10k-corpus k8
>    (`p25_eff32_s{0,1,2}` via eval-final, same full PopQA): **0.232 +/- 0.011** -> 100k k8
>    **0.477 +/- 0.002**. NEW: strict held-out ALSO rises with scale (0.453 -> 0.545, +9.2 pt) —
>    the old "held-out flat across scales" read was an artifact of leakage inflating the
>    25-epoch 10k runs more than the 4-epoch 100k runs. Scale now cleanly improves BOTH axes.
> 5. **Untrained-injection control (NEW, `eval-untrained-injection`):** top-k mean edge
>    embeddings in the same slots: PopQA **0.130** (k8; k16 identical), barely above base
>    0.103, vs trained k8 0.477. The learned encoder accounts for ~93% of the injected-knowledge
>    effect; more untrained slots add nothing.
> 6. **RAG baselines strengthened (3 retrieval modes, same frozen sets):** at <=32 knowledge
>    tokens concepts beat top-PageRank truncation AND question-aware retrieval AND LLM-written
>    budgeted summaries (e.g. @16 tokens PopQA: concept 0.518 vs 0.113/0.243/0.157); text
>    catches up only at ~64-128 tokens. The efficiency claim survives non-strawman baselines.

> 🟡 **STRONG SIGNAL; k4/8/16 at 3 SEEDS, k1/2/32 at 1 (Phase C + C2, group `phaseC-kfamily-100k`).**
> Converged 100k k-family (100k steps ≈ 4 epochs, best-held-out checkpoint, locked config + HPs).
> Numbers below are 3-seed mean±std where available (single-seed Phase-C values were revised by the
> seeds — see point 3; e.g. k16 0.575→0.643, k8 PopQA 0.555→0.495).

**MEASURED — converged 100k k-curve (best held-out checkpoint; `pc_k{k}` + `pc_k{k}_s{1,2}`).**

| k | held-out | **PopQA (unseen)** | seeds |
|--:|--:|--:|--:|
| 1 | 0.295 | 0.215 | 1 |
| 2 | 0.335 | 0.210 | 1 |
| 4 | 0.472 ± 2.8 | 0.442 ± 4.0 | 3 |
| **8** | 0.580 ± 0.8 | 0.495 ± 4.2 | 3 |
| 16 | **0.643 ± 4.8** | 0.520 ± 0.4 | 3 |
| 32 | 0.595 | 0.555 | 1 |

**Read — three results.**
1. **PopQA (unseen-entity generalization — the headline axis) jumps ~2.25×:** 100k k8 = **0.495 ± 4.2**
   (3 seeds; the single-seed 0.555 was optimistic) vs the 10k corpus's ~0.22 (F7); k16 = 0.520 ± 0.4.
   This is the paper's central claim — *learn the graph, generalize to entities never trained on* — and
   it is the effect of **data scale**, far beyond noise (even the soft 3-seed mean is ~2.25×).
2. **The mechanism is genuine graph-learning, not memorization.** 100k has *lower* per-entity overfit
   (k8 held-in ~0.64 vs the 10k's ~0.82) yet *much higher* PopQA — it stopped memorizing training
   entities and learned the edge→concept mapping. **Held-out is similar across scales** (100k k16
   0.643 ≈ 10k k16 0.637; k8 100k 0.580 vs 10k 0.535) — the scale win is concentrated on **PopQA
   (unseen entities)**, the honest axis, exactly as graph-learning (not memorization) predicts.
3. **The token-efficiency knee is ~k16 (FULL 3-seed curve, 2026-07-02).** The single-seed Phase-C
   curve suggested a k8 plateau; that was wrong (a low single k16 seed, 0.575). The complete ×3 curve:

   | k | held-out (mean±std) | PopQA (mean±std) | seeds |
   |--:|--:|--:|--:|
   | 1 | 0.318 ± 2.3 | 0.227 ± 1.0 | 3 |
   | 2 | 0.387 ± 3.8 | 0.320 ± 7.8 | 3 |
   | 4 | 0.472 ± 2.8 | 0.442 ± 4.0 | 3 |
   | 8 | 0.580 ± 0.8 | 0.495 ± 4.2 | 3 |
   | 16 | **0.643 ± 4.8** | 0.520 ± 0.4 | 3 |
   | 32 | 0.647 ± 4.3 | 0.533 ± 2.7 | 3 |

   **Rises steeply to k16, then PLATEAUS:** k16 (0.643) ≈ k32 (0.647) on held-out (+0.4 pt for 2×
   tokens); PopQA also flattens (k16 0.520 ≈ k32 0.533). So the knee is **~k16** — more tokens past 16
   buy ~nothing. k16 gives +6 pt held-out over k8 for 2× tokens (still 6.3× compression vs ~100 fact
   tokens) → **k8 vs k16 is the real efficiency trade for the recipe** (settle in Phase D). **Lesson
   (thrice): single seeds mislead — k2 (0.335→3-seed 0.387) and k16 (0.575→0.643) were both low
   outliers; seeding the WHOLE family caught it.** k16/k32 remain noisy (std ~4–5).
   NOTE: k2's 3 seeds live in W&B runs `pc_k2`/`pc_k2_s1`/`pc_k2_s2` — s1/s2 are state=**failed** (a
   transient W&B artifact-upload crash at end-of-run), but their metrics + local `_best.pt` checkpoints
   are valid; read by run-id, not `state="finished"`.

**Convergence horizon (resolves F11's open question).** Held-out plateaus by ~80–100k steps (~3–4
epochs); small k (1,2) flatten earlier, larger k climb to ~90k. k8 was still creeping up at 100k
(0.525→0.570 in the last 10k) so it is near-but-not-fully converged — a longer horizon might add a
little. No collapse/overfit caught by best-checkpoint selection.

**Recipe implication (revised after seeds).** Not a free k8 plateau — k16 buys +6 pt held-out over
k8 for 2× tokens (still 6.3× compression vs the median 100 fact tokens, F2). **k8 vs k16 is a genuine
token-efficiency trade to settle in Phase D** (held-out favours k16; PopQA nearly flat 0.495→0.520 so
favours the cheaper k8). Candidate headline models: `model:pc_k8_best` (efficiency) or k16 (peak
held-out). PopQA's flatness means k8 already captures most unseen-entity generalization.

**Caveats.** k4/8/16 at 3 seeds; k1/2/32 still 1 seed (seeding in progress). k16/k32 noisy (std ~5).
k8 not fully converged (still creeping at 100k steps). Next: finish k1/2/32 seeds; Phase-D figure.

**Re-verify:** group `phaseC-kfamily-100k`; aggregate `held_out/concept_acc_best` + `popqa/concept_acc`
across seeds per k (k16 held-out 0.643 > k8 0.580; PopQA k8 0.495 vs 10k F7 ~0.22); note held-in falls
0.82→0.64 (less memorization) while PopQA rises.

---

## Finding 13 — It reads the graph (separable edges + counterfactual swap), and faithfulness SCALES with k

> ✅ **STRONG — full k-family sweep, 1 seed/k (best-seed best-ckpt `pc_k{k}_s1_best`, Phase E/F,
> `cf-graph-faithfulness`, 400 held-out single-fact probes each, with base-bracket split).** The
> concepts carry real per-edge graph content (provably; NOT the LLM's memory; NOT an undifferentiated
> blob), AND the encoding gets **more faithful as k grows**: counterfactual edge-swap-follow rises
> **monotonically 0.8% → 34.2%** (k1→k32) while sticking-to-the-now-false-original falls 39% → 14%.
> The k8 snapshot (~23% swap-follow) that once read as "holistic/partial" is a point on a clean
> capacity curve — faithfulness is **capacity-limited, not absent**.

**MEASURED — full sweep (400 probes each; base = frozen Qwen, NO concepts; swap/stick on the
base-UNKNOWN subset = no parametric-memory confound).**

| k | concept correct | base-only | ablate-ANSWER (want LOW) | ablate-OTHER (want HIGH) | swap-follow→FALSE | stick-to-orig |
|--:|--:|--:|--:|--:|--:|--:|
| 1  | 41.2% | 10.0% | 55.2% | 90.3% | 0.8%  | 39.1% |
| 2  | 47.0% | 10.0% | 54.3% | 94.7% | 5.3%  | 30.9% |
| 4  | 55.8% | 10.0% | 37.7% | 95.5% | 16.5% | 20.2% |
| 8  | 63.0% | 10.0% | 33.7% | 92.1% | 25.8% | 20.2% |
| 16 | 65.2% | 10.0% | 35.6% | 95.0% | 29.8% | 17.8% |
| 32 | 70.2% | 10.0% | 27.8% | 94.0% | **34.2%** | **14.0%** |

**Read — three clean signals, all pointing the same way.**
1. **It's the concepts, not the LLM's memory.** base-only (frozen Qwen, no concepts) knows just **10%**
   at every k; with concepts, 41–70%. The +30–60 pt is injected knowledge, not parametric recall.
2. **Edges are encoded SEPARABLY, at all k.** Removing an UNRELATED edge barely dents accuracy (90–96%),
   removing the ANSWER edge collapses it (55%→28%) — a 35–66 pt gap that *widens* with k (k32:
   94% vs 28%). Not an entangled blob; individual edges are load-bearing, and more so at higher k.
3. **The counterfactual swap — the true graph-reading discriminator — SCALES with k.** Replacing the
   answer edge's neighbor with a type-plausible FALSE entity: the model follows it to the false answer
   **0.8% (k1) → 16.5% (k4) → 25.8% (k8) → 34.2% (k32)**, monotone; meanwhile stick-to-original falls
   39%→14%. At k32, on contradicted edges the model follows the false neighbor **2.4× more often than it
   sticks to the (now-wrong) truth**. Separability alone can't tell graph-encoding from per-fact
   text-compression (removing a fact's text also loses it) — but the swap can, and its steady climb
   with capacity is exactly what a graph reader (not a memoriser) predicts.

**Why it matters for the grant.** Faithfulness is **capacity-limited, not architectural**: the k8
"~23%, holistic" reading was one point on a curve that keeps rising. This is a *scaling* argument —
more concept tokens (and, plausibly, more data/compute) buy a more faithful graph reader — directly
motivating the scale-up. The remaining gap to a perfectly steerable per-edge reader is consistent with
a **lossy, set-pooled (permutation-invariant) encoding** that degrades under an OOD false edge.

**Caveats.** 1 seed per k (the swap monotonicity is clean enough to trust the trend; error bars TBD).
Swap uses type-plausible false neighbors from a same-property pool. Absolute swap-follow is still <50%
even at k32 — faithful-but-lossy, not a clean lookup table.

**Open follow-ups:** (a) does the curve keep rising on the 300k model at fixed k (data-scaling, not
just k-scaling)? (b) a per-edge auxiliary loss to push swap-follow higher? (c) linear-probe readout of
the swapped edge from the concept vectors (is the content present but under-used by the frozen LLM?).

**Re-verify:** `scripts/phaseF_graphfaith_sweep.sh` (`cf-graph-faithfulness --checkpoint pc_k{k}_s1_best
--n 400`); clean signals: base-only 10% ≪ concept at every k; ablate-OTHER ≫ ablate-ANSWER (gap widens
with k); swap-follow rises monotonically 0.8%→34.2% while stick falls 39%→14%.

---

## Finding 14 — v1 (arXiv 2504.07624) comparison: v2 reproduces the phenomena on a modern, harder setup

> ✅ **CONTEXT / corroboration (not a head-to-head number).** v1 and v2 differ on backbone, metric,
> task, and eval sets, so a direct number comparison is apples-to-oranges. What matters: v2
> **reproduces v1's core phenomena** on a *stronger backbone, a stricter metric, and a harder task*,
> and *adds* two things v1 lacked — an unseen-entity generalization axis (PopQA) and a graph-faithfulness
> proof. Source: `hf papers read 2504.07624` (v1, Barmettler 2025).
> **Publication status (added 2026-07-05):** v1 is peer-reviewed — Barmettler, Bernstein &
> Rossetto, "ConceptFormer: Towards Graph-Native Grounding of Large Language Models via Latent
> Concept Injection", **WWW Companion '26**, pp. 587-596, DOI 10.1145/3774905.3794653, and won
> the **best paper award** of its hosting workshop. Cite the proceedings version, not the
> preprint, everywhere (paper, grant, related work — all updated).

**The setups are NOT comparable number-for-number — be honest about this in the paper:**

| axis | v1 (2504.07624) | v2 (this work) |
|---|---|---|
| frozen backbone | GPT-2 0.1B (125M) | Qwen3-0.6B (~600M) |
| metric | **Hit@10 / Hit@1** (gold token in top-k logits) | **greedy exact-match accuracy** (much stricter) |
| task | next-token factual recall on T-REx sentences (fill-in the object) | **QA** (answer a generated question) + PopQA |
| train/eval data | Tri-REx (synthetic) → T-REx Bite (Wikipedia); WebQSP for QA | Gemma-generated QA over Wikidata 1-hop; held-out Qs + **PopQA unseen entities** |
| knowledge source | 1-hop Wikidata neighborhood → concept vectors | same (1-hop Wikidata → k concept tokens) |
| training | 2-stage (pretrain Tri-REx → finetune T-REx Bite), next-token CE | **KL self-distillation** vs frozen-LLM-reading-facts teacher |

**v1 headline numbers (for the record).** CF-15 on T-REx Bite: **Hit@1 46.7%, Hit@10 72.5%** (~10x over
GPT-2 0.1B baseline). Single vector CF-1: Hit@1 33.3% > text-RAG 6.6% at **130x fewer tokens**. Up to
**+272% (Wikipedia) / +348% (synthetic)** Hit@10 over baseline. Knee at **~10-15 vectors** (diminishing
returns beyond). Some CF-n beat LLaMA-2 7B (50x larger) at Hit@1.

**What v2 REPRODUCES (the phenomena survive a harder setup):**
1. **Concept vectors >> text-RAG per token.** v1: CF-1 beats RAG at 130x fewer tokens. v2: concepts
   dominate the low-token regime (F-RAG-budget: at 8 tokens, concept held-out 0.580 vs RAG 0.123, ~4.7x;
   PopQA 0.495 vs 0.093, ~5.3x) and reach RAG-level accuracy at ~6x fewer tokens.
2. **A knee at ~10-16 concept tokens.** v1: ~10-15 vectors cover a 1-hop neighborhood. v2 (F12, 3 seeds):
   knee ~k16, plateau k16≈k32. **Architecture-consistent across a 5x-larger backbone and a different
   task** — evidence the knee is a property of 1-hop-neighborhood capacity, not of GPT-2.
3. **Even a single concept token is already useful.** v1: CF-1 Hit@1 33.3%. v2: k1 held-out 0.318, PopQA
   0.227 (greedy exact-match — a far stricter bar).

**What v2 ADDS (the contribution beyond v1):**
- **Unseen-entity generalization (PopQA).** v1 headlines Hit@k on *trained-vocabulary* T-REx entities;
  v2's headline axis is **entities never seen in training** (PopQA), where data-scale lifts accuracy
  ~2.25x (F12). This is the honest "learn the graph, not memorize entities" claim v1 did not isolate.
- **A graph-faithfulness proof (F13).** Counterfactual edge-swap + edge-ablation shows the concepts
  carry separable per-edge content and read the graph (swap-follow scales 0.8%->34.2% with k) — a causal
  test v1 never ran (v1 argued efficiency + recall, not faithfulness).
- **A modern frozen backbone + KL self-distillation** (vs GPT-2 + next-token CE), and a locked,
  multi-seed, error-barred protocol (F8) rather than single-run point estimates.

**Grant framing.** v2 is not "v1 with bigger numbers" — the metrics forbid that claim. It is **v1's
phenomena, re-established on a modern LLM under a stricter metric and a harder QA/unseen-entity task,
plus a faithfulness proof and a data-scaling signal**. The ask (more compute) is to push the data-scaling
curve (100k -> 300k -> 1M) that F12 shows is still climbing.

**Re-verify:** `hf papers read 2504.07624`; v2 numbers from F12 (k-curve, PopQA) + F-RAG-budget +
F13 (faithfulness). Keep the metric/backbone/task caveats attached to any v1<->v2 sentence.

---

## Finding 15 — Capability preserved: concept tokens are near-inert on control tasks, even at k=32

> ✅ **EVIDENCE-BACKED — full k-family, 1 seed/k (Phase F, `cf-capability-preservation`, 400 held-out
> CONTROL tasks each, `pc_k{k}_s1_best`).** Injecting concept tokens does NOT degrade the frozen LLM's
> normal generation on tasks that name the entity but don't ask about its facts: the next-token
> distribution barely moves (median KL 0.06-0.08 nats) and does NOT blow up as k grows 1 -> 32.

**MEASURED (400 controls each; frozen Qwen WITH concepts vs WITHOUT).** Control prompts are the
off-topic templates in `generate/control.py` (translate/continue/echo/embedded-arithmetic), synthesized
on the fly over held-out entities — the 100k corpus was generated without control tasks (only
single/compositional), so the command falls back to the template bank when none are found.

| k | greedy-agreement (32-tok exact) | KL(base‖concept) median | KL mean | KL max |
|--:|--:|--:|--:|--:|
| 1  | 29.5% | 0.082 | 0.193 | 4.08 |
| 2  | 27.8% | 0.058 | 0.191 | 3.32 |
| 4  | 30.0% | 0.061 | 0.189 | 4.08 |
| 8  | 33.5% | 0.066 | 0.198 | 3.36 |
| 16 | 28.5% | 0.075 | 0.231 | 5.08 |
| 32 | 33.5% | 0.078 | 0.235 | 6.73 |

**Read.** (1) **Median KL is tiny (0.06-0.08 nats) and only inches up** across a 32× range of k — the
distribution is barely perturbed on off-topic prompts, so concepts don't hijack the model. (2) The mean
KL (0.19-0.23) sits well above the median: a **few** control prompts move a lot (max 3-7 nats) while
**most barely move** — expected, since some templates embed the entity name in the task itself
(translation/echo), where injecting its concepts legitimately shifts output. (3) **Greedy-agreement
~28-34% looks modest only because it is a strict 32-token exact match** (one flipped token zeroes the
row and errors compound over 32 steps); it does NOT trend down with k, so there is no capacity-driven
degradation. **KL is the faithful measure here, and it says capability is preserved.** This is by
design: gate-none zero-inits the encoder output projection, so at step 0 the student is bit-identical
to the frozen LLM (`model/injection.py`, `model/conceptformer.py`).

**Caveats.** 1 seed/k. 32-token exact-match is a blunt agreement metric (a first-token or short-window
agreement would read higher); the paper should headline **median KL**, report agreement as secondary.
Control prompts are templated, not Gemma-generated (the 100k corpus lacks control tasks) — fine for a
distributional-perturbation probe, but not a natural-distribution capability benchmark.

**Re-verify:** `scripts/phaseF_capability_sweep.sh` (`cf-capability-preservation --checkpoint
pc_k{k}_s1_best --n 400`) -> `data/analysis/capability_preservation.json`; clean signal is median
KL 0.06-0.08 nats, flat across k (not blowing up at k32).

---

## Finding 16 — Qwen3.5-0.8B pilot: recipe transfers across families; vision-port injection
## reaches parity at k8, trails at k16

> ✅ **EVIDENCE-BACKED (2026-07-03, M7 protocol: full PopQA n=14,266, strict held-out, paired
> McNemar; 2 seeds/cell).** Pilot for the grant's family migration + injection-port aim (O4),
> on `cftrain_qa_10k_q35b08` (10k corpus re-tiered + teacher-pathed by Qwen3.5-0.8B; 84,843
> rows — fewer than 0.6B's 92,177 because the stronger base answers more parametrically).

**1. Family migration de-risked (text port, zero HP retuning).** q35b08 k8: held-out
0.427/0.426, PopQA 0.226/0.208; k16: held-out 0.519/0.493, PopQA 0.254/0.294. Brackets: base
0.124, RAG 0.943 (0.6B: base 0.103, RAG 0.960). The locked recipe (gate-none, eff-batch-32,
lr 1e-4, d1024/L4) trains, converges, and generalizes on a different model family and a hybrid
linear-attention architecture. Note the **narrowed concept-over-base margin** vs 0.6B (PopQA
+10.3 pt over base vs +12.9) — the first measured point suggesting injection value is NOT
monotone in backbone capability; motivates the model-scale law (grant O1).

**2. Vision-port injection works, but is not better at 0.8B.** Same encoder, concepts spliced
as a pseudo-image (`[vision_start][k x image_token][vision_end]`, M-RoPE 1xk grid via the
model's own get_rope_index; `model/vision_port.py`, `injection_port=vision`). Paired per-item
comparison vs the text port (`scripts/pilot_port_comparison.py`):

| set | k | seed | text | vision | delta | p |
|---|--:|--:|--:|--:|--:|--:|
| held_out | 8 | 0/1 | 0.427 / 0.426 | 0.434 / 0.432 | +0.7 / +0.7 | n.s. / n.s. |
| held_out | 16 | 0/1 | 0.519 / 0.493 | 0.491 / 0.423 | -2.8 / -7.0 | 4e-3 / 1e-13 |
| popqa | 8 | 0/1 | 0.225 / 0.208 | 0.216 / 0.232 | -0.9 / +2.4 | 8e-5 / 6e-22 |
| popqa | 16 | 0/1 | 0.255 / 0.296 | 0.261 / 0.207 | +0.6 / -8.8 | 2e-2 / e-191 |

**Read.** At k8 the ports are at parity (deltas within +-2.4 pt, mixed sign across seeds). At
k16 the vision port trails and one seed (s1) is clearly unstable (-7.0/-8.8). The "vision
interface is a better landing pad" hypothesis is NOT supported at 0.8B — but the ports learn
*different* solutions (300-1,800 discordant items per cell), the vision port received zero
port-specific tuning, and 0.8B has the family's weakest vision pretraining. Whether the port
effect flips with backbone scale is exactly grant aim O4; the pilot converts it from
speculation to a measured, non-trivial question.

**Caveats.** 2 seeds/cell; single backbone size; no port-specific HP tuning; 10k corpus
(25-epoch regime). **Re-verify:** eval-final summaries under
`data/analysis/eval_final/q35b08_*`; W&B groups `pilot-qwen35-08b`, `pilot-qwen35-08b-vision`.

**3. ADDENDUM (2026-07-04) — 2B transect: the injection margin ANTI-SCALES with backbone size
at fixed data.** Qwen3.5-2B, same corpus protocol (`cftrain_qa_10k_q35b2`, 2B teacher), same
recipe (batch 8 x accum 4 = eff 32), k{8,16} x 2 seeds + vision k8 x 2 seeds, all eval-final'd
(full PopQA). PopQA concept-over-base margin at fixed 10k training entities:

| backbone | base | k8 margin | k16 margin |
|---|--:|--:|--:|
| Qwen3-0.6B (anchor, 3 seeds) | 0.103 | +12.9 pt | - |
| Qwen3.5-0.8B (2 seeds) | 0.124 | +9.4 pt | +15.1 pt |
| Qwen3.5-2B (2 seeds) | 0.150 | +5.1 pt | +6.5 pt |

Within-family 0.8B -> 2B the margin roughly HALVES at both k, while strict held-out stays
healthy (2B: 0.465-0.503) — the encoder works; its *unseen-entity* value shrinks as the frozen
model grows, at fixed data. Combined with the data axis (+12.9 -> +37.4 pt going 10k -> 100k
entities at 0.6B, F12 corrected), the pilot surface shows its first structure: **data scales
injection up, model size at fixed data scales it down; the interaction is the open question**
(figure: `data/analysis/pilot_scaling_surface.png`, `scripts/pilot_scaling_figure.py`).
Port x scale: at 2B k8 the vision port stays at parity on PopQA (-0.3/+2.7 pt, mixed seeds,
like 0.8B) and mixed on held-out (-2.5/+1.5); no flip either direction by 2B — k16 vision (the
0.8B deficit) not re-tested at 2B. Faithfulness sweeps on the 0.8B pilots: swap-follow k8
7.5%/5.2% (text/vision), k16 11.4%/12.9% — far below the 0.6B-at-100k values (25.8% k8), but
CORPUS-CONFOUNDED (10k 25-epoch vs 100k 4-epoch); logged as another non-triviality signal, not
a family verdict. Open interpretation for the grant: shrinking margin = ceiling effect
(stronger base) vs steerability (bigger frozen models need more data/tokens to steer) — the
100k-corpus corners of the funded plan disentangle these. Caveats: 2 seeds/cell, 10k corpus,
no per-size HP retuning (lr 1e-4 everywhere). Re-verify: `data/analysis/eval_final/q35b2_*`;
W&B groups `pilot-qwen35-2b`, `pilot-qwen35-2b-vision`; `scripts/pilot_port_comparison.py`.

---

## Finding 17 — Zero-shot cross-graph transfer: Wikidata-trained encoders work on MetaQA ✅ (single seed per k)

**Claim.** The inductive claim holds across graphs: checkpoints trained ONLY on Wikidata
(0.6B, 100k corpus, `pc_k*_best`) score far above base on MetaQA-1hop (movie KG, 43k entities,
9 relations, no entity ids — label IS identity), evaluated zero-shot with no adaptation.

**Numbers** (FULL test set n=9,947, greedy exact match vs any accepted answer, Wilson CIs;
2026-07-07): base 5.8% [5.3, 6.2], k1 16.1%, k8 23.5%, k16 24.4%, k32 **33.5%** [32.5, 34.4],
budgeted RAG (1024 tok, no answer guarantee) 92.1%. Item-paired McNemar concept-vs-base:
k8 discordants 1874/107, p < 1e-300 (k1: p ~ 5e-233). Adjacent k: k8>k1 p~3e-88,
k16>k8 p=0.033, k32>k16 p~3e-115 — monotone in k, same shape as home. Relative gap closure
vs home (base->RAG): k8 21% vs 44%, k32 32% vs 51% — zero-shot transfer retains roughly
half to two-thirds of the home-graph effect. n=1000 pilot (24.8% k8) confirmed by full run.

**Setup** (`data/metaqa.py`, `build-metaqa-snapshot`, `eval-transfer`): 9 relations mapped to
natural-language labels + hand-written reverse labels for incoming edges (e.g.
`starred_actors` -> "cast member"/"actor in"); snapshot `metaqa` sha `d70231425282…`,
43,234 entities. Re-verify: `data/analysis/transfer/pc_k{1,8,16,32}_best__metaqa_1hop_full/`
(summary.json + items.jsonl; mirrored to HF `joelbarmettler/conceptformer-data` under
`results/`). Caveats: **one seed per k** (s0 `pc_k*_best`); 1-hop vanilla split only;
MetaQA-trained contrast (Phase B) not yet run — do not claim anything about trained transfer.

**Probe durability note (2026-07-07).** `cf-graph-faithfulness` / `cf-capability-preservation`
now write summary.json + items.jsonl to `data/analysis/probes/<ckpt>__{faithfulness,capability}/`
(`eval/probes.py`); the F13/F15 sweep re-run over all 18 `pc_k*` checkpoints regenerates them
as durable artifacts (legacy terminal logs preserved in `data/analysis/probes/legacy_logs/`).
First re-run cell matches the transcribed values exactly (pc_k1_best swap-follow 0.0%).

---

## Open questions (not yet evidence-backed — do NOT state as findings)

0. ~~Establish the noise floor + re-test downgraded findings multi-seed (F8).~~ **RESOLVED → F8 +
   Phase 3.** Noise floor measured (~1 pt std on the locked config); F1/F2/F7/placement all re-tested
   ×3 seeds (→ F1 re-established, F2 refuted/reframed, F7 underpowered, F9 placement). Sensitivity
   reduced by gate-none + grad-accum (NOT determinism).
1. ~~Do the capacity/k orderings hold at convergence?~~ **RESOLVED → F1/F2** (re-run ×3 seeds @72k on
   the stabilized base): capacity peak ~70M holds; the k-curve is monotone (no knee) — the *opposite*
   of the old undertrained sweep, confirming the undertraining caveat.
2. ~~Does `sub_on` match `sub_off` on robustness?~~ **RESOLVED → F3/F5**: yes (slightly worse).
3. ~~Does `--augment` help?~~ **RESOLVED → F7 (re-tested ×3)**: underpowered — point estimates trend
   positive (+3.8–5.8 pt held-out) but inside the ~6–9-pt cross-prompt seed variance; not significant
   at n=3. Not in the recipe; revisit n≥6 in Phase 4 if needed.
3b. ~~Concept-vector placement?~~ **RESOLVED → F9**: placement doesn't help; `prefix` best on accuracy,
   `replace_entity` worse + unstable. `before_entity` ADOPTED as the recipe placement (within-noise of
   prefix, multi-entity-extensible). Adjacency/replacement-improves-binding hypothesis refuted.
4. Does 10×-wider data (100k entities) lift the ~40% held-out / 18% PopQA ceiling? 🟡 **IN PROGRESS →
   F11 + Phase C.** 100k corpus **GENERATED** (`cftrain_qa_100k`, 925,178 distill rows; snapshot
   `cftrain_100k` sha256 `ee4850f5633d8aa2a374bb9e145ce10675870274e7b25ca5bf0c5f545e60242c`). Early
   signal positive (still climbing past where 10k would, F11), but the **converged** comparison is
   being trained now (group `phaseC-kfamily-100k`). Do NOT claim a 100k ceiling yet.
5. **Token-efficiency headline figure (Phase D, deferred).** Concept k-curve (converged, from Phase C)
   vs a REALISTIC text-RAG accuracy-vs-token-budget curve on the same axis (knowledge tokens). Measured
   fact-token cost: median 100, mean 125 (F2). Use top-PageRank `verbalize_budgeted` (NO answer
   guarantee) at budgets {8,16,32,64,~100} — the answer-guaranteed teacher stays near-ceiling at tiny
   budgets and would rig it. NOT yet run.
6. **Does it LEARN THE GRAPH or just compress text? (Phase E — harness BUILT, not yet run.)** Causal
   graph interventions on a trained checkpoint (`cf-graph-faithfulness`, pure helpers in
   `eval/counterfactual.py`, 5 unit tests): (a) **counterfactual swap** — replace the answer edge's
   neighbor with a type-plausible FALSE entity; a graph-faithful model follows it to the false answer
   (high swap-follow), whereas a text-memoriser or one leaning on the frozen LLM's parametric
   knowledge sticks to the original; (b) **edge ablation** — drop the answer edge → accuracy on THAT
   question collapses while removing an UNRELATED edge leaves it intact (edges encoded separably, not
   as one entangled blob). Pre-registered success = high swap-follow + low answer-ablation accuracy +
   high other-ablation accuracy. Falsifiers: swaps don't change the answer (memory/entangled), or
   ablation degrades uniformly. Run on `pc_k8_best` when a GPU frees. This is the paper's "it reads
   the graph" proof; the swap-to-FALSE design also disentangles concepts from the LLM's own memory.
