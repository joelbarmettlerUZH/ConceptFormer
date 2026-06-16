# ConceptFormer v2 — Research Findings (living log)

**Purpose.** A running record of experimental findings, written so every quantitative claim is
traceable to ground-truth data *before* anything goes into the paper. Numbers drift; this file is
the audit trail that lets us re-verify each one. **Do not cite a number from here in the paper
without first re-checking it at the linked source.**

Last updated: 2026-06-16.

---

## How to verify (anchors)

| Thing | Where the ground truth lives |
|---|---|
| W&B entity / project | `university-of-zurich` / `conceptformer-v2` |
| W&B run URL pattern | `https://wandb.ai/university-of-zurich/conceptformer-v2/runs/<run_id>` |
| W&B sweep URL pattern | `https://wandb.ai/university-of-zurich/conceptformer-v2/sweeps/<sweep_id>` |
| Model checkpoints | W&B **artifacts** `model:<checkpoint_name>` (e.g. `model:sub_off_72k`), attached to their run |
| Training corpus | `data/cf_train/cftrain_qa_10k/qa_distill.jsonl` — **92,177** distill examples |
| Snapshot (graph) | `data/snapshots/cftrain_10k` — 10,000 subgraphs, `min_edges=6` |
| Snapshot integrity | sha256 `8aa882a06d56ef028b0a2de99ac4c7f9c1b1f4dcc3c54dfd5a1ab978c93d633e` (in `manifest.json`) |
| Backbone (frozen) | `Qwen/Qwen3-0.6B` (teacher and student share it) |

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

## Finding 1 — Encoder capacity saturates early; bigger is not better

> ⚠️ **DOWNGRADED → within the init-noise floor (see F8).** Single-seed runs; the full spread is
> only 8 pts ≈ the ~±8-pt init noise, so the "bigger isn't better" ordering is **NOT established**.
> Needs a multi-seed re-run. Also subject to the 24k undertraining caveat above.

**Claim.** Held-out accuracy is flat-to-declining across encoder size: a ~21M-param encoder matches
a 70M one and **beats** a 231M one. Capacity is not the bottleneck.

**Source.** Sweep `642bhv2c` — <https://wandb.ai/university-of-zurich/conceptformer-v2/sweeps/642bhv2c>
(16 runs, grid `d_model ∈ {512,768,1024,1536} × n_layers ∈ {2,3,4,6}`, k=8, 24k steps, subsample on).

| d_model | n_layers | params | held_out/concept_acc | run_id |
|--:|--:|--:|--:|---|
| 1024 | 4 | 70.4M | **0.315** | `bdxi4ob3` |
| 768 | 2 | 21.3M | 0.305 | `qmaiv9mf` |
| 768 | 4 | 40.2M | 0.300 | `fcunbg0g` |
| 1536 | 3 | 118.1M | 0.300 | `m2ahnefj` |
| 1536 | 6 | **231.4M** | 0.270 | `itp2kb6z` |
| 512 | 3 | 14.2M | 0.235 (min) | `kg0alzho` |

Full 16-row table in the sweep. **Spread is only 0.235→0.315** (8 pts) across a 23× param range.
Sweet spot ≈ **d768, L2–4 (21–40M params)**.

**Re-verify:** open the sweep, sort runs by `held_out/concept_acc`, confirm the 231M run
(`itp2kb6z`) sits below the 21M run (`qmaiv9mf`).

---

## Finding 2 — k-curve: an entity's neighborhood compresses into ~4–8 concept tokens

> ⚠️ **DOWNGRADED → within the init-noise floor (see F8).** k=4→k=8 differs by ~3 pts ≪ the ~±8-pt
> init noise; the k=1/2 starvation (−6 pts) is more likely real but still single-seed. The knee
> location is **NOT established** without a multi-seed re-run. Also subject to the 24k caveat above.

**Claim.** Accuracy is starved at k=1–2, jumps at k=4, peaks at k=8, then flat/noisy. The knee at
**k≈4–8** is the compression headline. KL keeps falling past the knee while accuracy does not (KL ≠
task accuracy).

**Source.** Sweep `aenmzr4v` — <https://wandb.ai/university-of-zurich/conceptformer-v2/sweeps/aenmzr4v>
(6 runs, `k ∈ {1,2,4,8,16,32}`, capacity fixed d1024/L4, 24k steps, subsample on).

| k | held_out/concept_acc | popqa/concept_acc | held_out/val_kl | run_id |
|--:|--:|--:|--:|---|
| 1 | 0.250 | 0.160 | 0.859 | `8equtffh` |
| 2 | 0.250 | 0.160 | 0.862 | `qc830h57` |
| 4 | 0.285 | 0.195 | 0.802 | `58abpu0m` |
| **8** | **0.315** | **0.195** | 0.799 | `b6z1objc` |
| 16 | 0.280 | 0.155 | 0.825 | `gte0de89` |
| 32 | 0.310 | 0.190 | 0.783 | `bibpizex` |

**Re-verify:** open the sweep; confirm the k=8 run (`b6z1objc`) is the held-out max and that
`held_out/val_kl` at k=32 (`bibpizex`, 0.783) is *below* k=8 while accuracy is not higher.

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

> ⛔ **DOWNGRADED → NOT established (within init noise, see F8).** The augment effect **flips sign**
> across init pairs: old code aug-on−aug-off = +8.5 (`0ze8ia5q` 0.485 − `kuyslwjf` 0.400), new code
> = −6.5 (`xqui9aio` 0.400 − `uqdcag0a` 0.465). Mean ≈ 0 ± ~7.5. The cross-prompt robustness harness
> compared two SINGLE checkpoints, so it does not rescue the claim — `0ze8ia5q` may just be a lucky
> init. **Must be re-tested multi-seed before any augment claim.** Keeping the section below for the
> measured numbers, but the conclusion is suspended.

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

## Finding 8 — Large run-to-run variance (~8 pts) — cause is HYPOTHESIS, not yet proven

> 🟡 **PARTIAL — observation solid, attribution unconfirmed.** The *existence* of ~8-pt run-to-run
> variance is observed; that it is *caused by weight init* (vs CUDA nondeterminism or the old-vs-new
> code difference) is a **hypothesis under test**, not a diagnosis. It still downgrades F1/F2/F7
> because *whatever* the cause, single-run effects ≲8 pts aren't trustworthy.

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

---

## Open questions (not yet evidence-backed — do NOT state as findings)

0. **TOP PRIORITY — establish the noise floor + re-test downgraded findings multi-seed (F8).** Run
   key configs ×3 seeds (now that torch is seeded) to get mean±std, then re-decide F1 (capacity),
   F2 (k), F7 (augment), and the placement ablation against the noise band. Also probe whether the
   init sensitivity can be *reduced* (gate warmup / lower gate-LR / EMA) so fewer seeds are needed.
1. Do the capacity/k orderings hold at convergence (cached, ~60k+ steps)? (sweeps were 24k, undertrained)
2. ~~Does `sub_on` match `sub_off` on robustness?~~ **RESOLVED → F3/F5**: yes (slightly worse);
   subsample question fully closed.
3. ~~Does `--augment` help?~~ **RE-OPENED → F8**: the +7.7 in F7 flips to −6.5 on another init, so
   augment's effect is unproven (within noise). Must be re-tested ×3 seeds before any claim.
3b. ⛔ NOT YET EVIDENCE-BACKED — **Concept-vector placement.** Does *where* the k concept tokens sit
   in the user message matter? Modes: `prefix` (current baseline), `before_entity`, `after_entity`,
   `replace_entity` (entity surface form removed → concepts must fully substitute). Entity is
   verbatim in 100% of questions, so all modes run on the full corpus. Needs a `placement` config +
   per-example student head. Hypothesis: adjacency/replacement improves entity↔knowledge binding.
4. Does 10×-wider data (100k entities) lift the ~40% held-out / 18% PopQA ceiling? ⛔ NOT YET
   EVIDENCE-BACKED. Snapshot **built and staged** — `data/snapshots/cftrain_100k`, 100,000 usable
   subgraphs (261,475 candidates, 38% pass), sha256 `ee4850f5633d8aa2a374bb9e145ce10675870274e7b25ca5bf0c5f545e60242c`.
   Gemma QA generation + teacher extraction **NOT started** — deliberately gated on Q2/Q3 (robustness).
