# ConceptFormer v2 — orientation for Claude

Dense, code-free orientation. **Companions:** `TODO.md` (live plan), `docs/RESEARCH_FINDINGS.md`
(evidence log — every metric → W&B run id / artifact / repro command), `docs/MODEL_DESIGN.md` +
`docs/CONCEPTFORMER_V2_EXPLAINED.md` (the design rationale for a PhD reader). This file holds the
*why*, the *learnings*, the *pitfalls*, and *how to run* — not anything the code already states.

## Working discipline (read every session)
- **This ships as a paper; the code is open-source.** Hold everything to publication quality:
  reproducible, readable, tested, honest. Reviewers and readers will run this.
- **Evidence vs hypothesis — be ruthless.** Something is a *finding* only when shown **empirically**
  by a controlled experiment that clears the noise floor (≥3 seeds, mean±std). Until then it is a
  **hypothesis to be proven** — say so, and never let it enter the paper as a result. Tag every claim
  in `docs/RESEARCH_FINDINGS.md` (✅ evidence-backed / 🟡 partial / ⛔ not yet). One run is not
  evidence (see F8). When unsure, design the experiment that could *falsify* the claim.
- **W&B is the single source of truth for numbers.** Every metric you cite must be traceable to a
  **run-id** (or sweep/artifact) and **re-verified at source** before it goes in a doc or the paper —
  numbers drift, memory lies. Every checkpoint/result must be a **downloadable W&B artifact**
  (`model:<name>`); if a result isn't reproducible from W&B, it doesn't count.
- **Research before you design.** Before any architecture/training decision, check the literature
  with the **`hf` CLI papers** tools; record the **arXiv id in a code comment** at the site of the
  choice (e.g. why a resampler, why this gate, why these positions). Decisions cite prior art.
- **Keep the docs in sync — constant, deliberate effort.** *After each run or change*, explicitly ask
  whether `CLAUDE.md`, `TODO.md`, `docs/RESEARCH_FINDINGS.md`, or `docs/*` need updating, and do it.
  These four drift the moment you stop tending them; an out-of-sync doc is worse than none.

## What this is
Knowledge injection for a **frozen** LLM. For a Wikidata entity, encode its **1-hop neighborhood**
into `k` constant **soft "concept tokens"** and splice them into the LLM's prompt in place of
verbalized facts — so the model "knows" the entity without retrieval text or fine-tuning. Rebuild of
the author's published v1 (arXiv 2504.07624, Joel Barmettler). Framing is **transductive**:
precompute per-entity concept vectors offline; at inference you pay `k` tokens, not a RAG passage.
The paper's thesis is **token efficiency** (k≈8 soft tokens standing in for ~100–200 fact tokens) and
that the encoder learns the *graph*, not just context-compression, despite only ever seeing one
entity+neighborhood at a time.

Backbone is **Qwen3-0.6B, frozen** (teacher and student share it). The only trainable thing is the
ConceptFormer (encoder + gate), ~10–230M params depending on width/depth.

## Objective (the heart)
**KL self-distillation against the same frozen LLM.** Per example:
- **teacher** = frozen LLM reading `[system][verbalized facts as TEXT][question]`, greedy-decodes an
  answer path (stored offline).
- **student** = frozen LLM reading `[system][k concept tokens][question]`.
- Loss = per-position full-vocab KL between teacher and student distributions along the teacher's
  stored greedy path. Optimization gathers only the ~20 path positions and applies the LM head there
  (identical result, far less compute/memory). See `train/{trainer,forcing,losses}.py`.

**Eval brackets** (always reported together; `train/harness.py`, `eval/`): **base** = frozen LLM, no
knowledge (floor); **teacher/RAG** = frozen LLM reading facts as text (ceiling, ~99% on the train
corpus); **concept** = the system under test. Three accuracy axes: **held_out** = unseen *questions*
about *trained* entities (query-generalization); **held_in** = trained questions (overfit gauge);
**popqa** = unseen *entities* (entity-generalization — the hard, headline axis).

## Architecture (modules → purpose; read the file for the how)
- `model/backbone.py` (+ `model/chat.py`): wraps the frozen Qwen chat model; exposes embedding,
  `forward_hidden`, `lm_head`, label-embedding pooling. The asymmetry the design exploits: text facts
  are bounded by the context window; concept tokens are not.
- `model/featurizer.py`: an edge → feature vector = concat(property-label embedding, neighbor-label
  embedding), so `d_in = 2·d_llm`. Neighborhood = an **unordered set** of edge features (+ mask).
- `model/encoder.py` (`ConceptEncoder`, `ResamplerBlock`): a **Perceiver/Flamingo latent-query
  resampler**. `k` learned latent queries cross-attend over the edge set (permutation-invariant → a
  set, not a sequence) then self-attend; output projection maps internal `d_model` → `d_llm`.
  Encoder capacity (`d_model`,`n_layers`) is **decoupled** from the LLM width.
- `model/injection.py` (`ConceptGate` + splice/pack/position helpers): zero-init **per-token tanh
  gate** — at step 0 `tanh(0)=0` so concepts contribute nothing and the student is *bit-identical* to
  the frozen model (capability preservation). Concepts occupy a **contiguous, unit-step RoPE span**
  (no gaps/overlap/shared positions — all three are empirically harmful for a frozen RoPE model).
- `model/conceptformer.py`: composes encoder+gate. `gate_mode`: `"tanh"` (default) or `"none"`
  (drop the gate, zero-init the encoder output projection instead — a round-2 stabilization arm).
- `train/trainer.py`: the integration layer — featurize, build teacher/student sequences via a
  sentinel-placeholder split of the chat template (the VLM "image-token" trick), run the two frozen
  forwards, KL. Also: teacher-hidden caching, prompt augmentation, neighbor subsampling, EMA,
  gradient clipping, concept **placement**, and all eval/checkpoint logic.
- `verbalize.py`: facts-as-text for teacher/RAG. `verbalize_budgeted` keeps top-PageRank neighbors
  within a token budget; `verbalize_with_answer` *guarantees* the question's answer edge is present
  (+ optional per-epoch distractor shuffle).
- `data/`: the offline pipeline (`select`, `snapshot`, `wikidata`, `benchmarks`=PopQA, `quality`,
  `coverage`). `generate/`: Gemma-via-vLLM question generation (`gemma`, `dataset`, `prompt`,
  `schema`, plus `descriptive`/`control`/`grounding` task variants). `eval/`: PopQA-style scoring,
  token accounting, predictors. `schemas.py`/`config.py`/`cache.py`: pydantic models, settings,
  on-disk caches.

## Levers on the model/training (all in `TrainConfig` / `cf-train` flags)
`k` (concept tokens), `d_model`/`n_layers` (encoder capacity), `lr` + `gate_lr` (gate needs a **100×
higher LR ~1e-2** or it sits near 0 and starves the encoder — a real dead-zone), `temperature`,
`augment` (distill under 5 system prompts, eval under a held-out 6th → prompt-agnostic vectors),
`subsample` (re-sample teacher distractors/step — **no benefit at the current budget; see F3**),
`placement` (`prefix`|`before_entity`|`after_entity`|`replace_entity` — where concepts sit vs the
entity mention; `replace_entity` removes the surface form so concepts must *be* the entity),
`cache_teacher` (precompute teacher path-hiddens → ~2× faster; incompatible with subsample),
`grad_clip`, `ema_decay`, `gate_mode`, `seed`.

## Data pipeline (offline; `cli.py` commands, artifacts under `data/`)
`select-entities` (popularity-stratified from the danker PageRank dump, PopQA subjects excluded) →
`build-cftrain-snapshot` (fetch 1-hop neighborhoods from Wikidata, filter junk/thin/unlabeled until
`target` *usable*; ~38% pass rate) → `generate-cftrain` (Gemma/vLLM writes QA: single +
compositional + descriptive + control) → `tier-cftrain` (label answerability vs the budgeted facts)
→ `extract-teacher-paths` (store the frozen teacher's greedy answer path per example). PopQA snapshot
built separately for the entity-generalization eval. Each snapshot's `manifest.json` carries an
integrity `sha256`.

## How to run
- Everything via **uv**: `uv run --group infer python -m conceptformer.cli <cmd> [flags]`. The
  `infer` group pulls torch/transformers/wandb (heavy); the bare group is the lightweight data layer.
- Train/generalization: **`cf-train`** (the workhorse — held-out question split, all metrics, W&B,
  artifact, optional PopQA). Sweeps: **`cf-sweep`** (W&B sweep; `--params "name=v1,v2;..."`, one agent
  per GPU). Robustness: **`eval-prompt-robustness --checkpoint <name>`** (scores a checkpoint across
  many system prompts; auto-pulls the W&B artifact if the local file is gone). Debug: `cf-overfit`.
- **GPUs:** two 24 GB cards. One training run uses **one** GPU (teacher+student share it). Run two
  independent experiments by pinning `CUDA_VISIBLE_DEVICES=N` **and passing `--device cuda:0`**
  (CVD remaps the visible card to ordinal 0 — passing `--device cuda:N` is the classic bug → "invalid
  device ordinal"). Always set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. **`batch 64`
  OOMs**; `batch 32` fits. Larger *effective* batch ⇒ **`--grad-accum N`** (effective = `batch*N`,
  peak memory stays at one batch; LR left unscaled so batch size is a clean lever). Effective batch
  **32 is the sweet spot** at 72k (eff64 overshoots — see F8); the locked base uses `batch16 x accum2`.
- **W&B:** entity `university-of-zurich`, project `conceptformer-v2` (auth via `~/.netrc`). `cf-train`
  auto-logs full trajectories + a **`model:<checkpoint>` artifact**. Read runs by **run-id or
  `state="finished"`**, never by name alone (killed runs leave same-named `crashed`/`failed` ghosts).
- **Gate:** `uv run pytest -q -m "not integration"`, `uv run ruff check`, `uv run ty check`. Keep all
  three green; line length 100; ASCII only in source (RUF flags en-dash/×/≈).
- **Background jobs:** prefer the harness's `run_in_background` (it notifies on completion). `nohup …
  &` *detaches* and the harness thinks it finished instantly — only use it for fire-and-forget, and
  add a separate tracked watcher. To stop a sequential chain after the current run, `kill` the **loop
  PID only** (single signal, not the group) — the in-flight `cf-train` child reparents and finishes.

## Where things live
`data/` (git-ignored, machine-local): `cf_train/<dataset>/` (qa, qa_tiered, qa_distill jsonl +
manifest), `snapshots/<name>/` (subgraphs.jsonl + manifest+sha), `checkpoints/*.pt`,
`wikidata_cache.sqlite` (~1.4 GB — makes snapshot rebuilds cheap), `generations.sqlite`.
**Checkpoints also live on W&B as `model:<name>` artifacts** (durable; the local dir is not). Corpus
itself is *not* yet a W&B artifact — reproduce from the snapshot sha if needed. Datasets in play:
`cftrain_qa_10k` / snapshot `cftrain_10k` (the workbench, ~92k distill examples, 10k entities);
`cftrain_100k` snapshot **built & staged** but generation **gated/not started**. Branch
`conceptformer-v2`; commits authored `Joel Barmettler <joel.barmettler@uzh.ch>`, trailer
`Co-Authored-By: Claude Opus 4.8 (1M context)`. `memory/` is intentionally **not** committed.

## State of knowledge (summary — details + run-ids in `docs/RESEARCH_FINDINGS.md`)
**F8 (run-to-run variance) — LARGELY RESOLVED, config LOCKED.** The original problem: ~8-pt held-out
outcome variance at 72k for "identical" configs, making effects un-steerable. Seeding torch made
*init* reproducible but same-seed runs still diverged → sensitivity-driven, **fixed in
training/architecture, not by forcing CUDA determinism**. The fix that emerged from a controlled
search (round-2 arms EMA/gate-none/grad-clip/batch, then a 72k batch-size curve): **gate-none**
(drop the saturating tanh gate, zero-init the encoder out-proj instead) **+ effective batch 32 via
`--grad-accum 2`**. Measured @72k, ×3 seeds: **held-out 0.535, std 1.08 pt, range 2.5 pt** — the
tightest of every config, and highest accuracy. Curve is **non-monotone: eff32 is the sweet spot,
eff64 overshoots** (mean drops, spread widens). EMA *hurt*; batch alone (true b32) was accurate but
unstable (std 3.06). This is the **locked Phase-3 base** (`p25_eff32_*`, group `phase25-grad-accum`).
Caveat: n=3 throughout. The downgrades stand until re-tested: **F1 (capacity saturates), F2
(k-knee≈4–8), F7 (augment +7.7)** were single-seed, inside the old noise — Phase 3 re-runs them on
the locked base with error bars. What survives independently: **F3** (neighbor-subsample = no
benefit, ~2× slower → cached teacher), **F4** (24k is *undertrained*; converges ~40%+ held-out by
~54–72k then overfits), **F6** (at the 2048-tok budget the answer edge is ~always present).

**Established protocol going forward:** subsample **off** (cached), gate-none + grad-accum2 base,
**≥3 seeds, report mean±std**, claim an effect only if it clears the ~1-pt floor. Current phase
(Phase 3): re-run capacity (3.1, running) → k-curve at best capacity (3.2) → augment (3.3) →
placement (3.4) on the locked base; update findings with error bars (3.5); assemble best-model
recipe (4.1); decide the 100k-entity scale-up (4.2). Live plan + status in `TODO.md`.

**M7 eval overhaul (2026-07-02) — supersedes all pre-M7 accuracies.** Eval sampling is now
decoupled from the training seed (`eval/evalsets.py`, fixed `EVAL_SAMPLE_SEED`; smaller samples
are prefixes of larger ones → everything pairs); held-out splits group by **fact** (paraphrase
leakage was 32.8%!); definitive numbers come from **`eval-final`** (FULL PopQA n=14,266 + strict
held-out, Wilson CIs, per-item dumps) with paired tests in `eval/stats.py`; baselines:
`eval-untrained-injection` (near-floor → the encoder is the effect) and 3-mode
`cf-rag-budget-curve`. Corrected headlines: k8 PopQA **0.477±0.002** (base 0.103, RAG 0.960),
scale-up 10k→100k **2.05×** on identical sets, k-curve monotone through k32 (old "k16 plateau"
was n=200 noise), old 4–8-pt seed spreads were mostly eval sampling noise. Positioning + niche:
`docs/RELATED_WORK.md` (closest: xRAG/KBLaM/Knowledge-Prompts/GNP; grant lead = scaling laws of
knowledge injection). Never quote a pre-M7 number in the paper.

## Coding guidelines (publication-grade open source)
- **Gate is non-negotiable:** `ruff check`, `ty check`, and `pytest -q -m "not integration"` all
  green before any commit. Line length **100**; **ASCII only** in source (ruff RUF flags en-dash, ×,
  ≈, etc.). The ruff rule set (E/W/F/I/UP/B/SIM/RUF/ANN/RET/C4/PTH) is in `pyproject.toml` — respect
  it rather than widening ignores.
- **Type everything.** Full annotations; `ty` clean (no silencing). Narrow `X | None` explicitly
  (raise on `None`) rather than blanket asserts. CLI options/args use **typer `Annotated[...]`** with
  a `help=` string. Prefer `Sequence`/protocols over concrete containers at boundaries.
- **Comments explain WHY, never WHAT.** The code already says what it does; a comment earns its place
  only by giving the rationale, the trade-off, the gotcha, or the prior-art (arXiv id). No narration.
  Match the surrounding file's comment density and idiom — write code that reads like its neighbours.
- **Docs contain no code** (they drift); reference files/symbols instead. Keep functions small and
  **extract pure helpers** so logic is unit-testable without the GPU/model (e.g. `place_concept_slot`,
  the verbalizers, `gate_mode` invariants are tested this way). Add a test with every behavior change.
- **Determinism of *intent*:** seed what you can, but the goal is robust training (low across-run
  spread), not papering over chaos with `use_deterministic_algorithms` — fix sensitivity in the
  model/training, not by pinning the hardware (see F8).

## Pitfalls (hard-won; do not relearn)
- **Don't trust a single run** for any effect ≲8 pt — it's inside the noise floor (F8). Multi-seed.
- **Don't call generalization/saturation off undertrained runs.** 24k was undertrained; "ceilings"
  moved a lot by 54–72k. Add a held-in readout to tell *overfit* from *undertrained*.
- **KL ≠ accuracy:** runs reach near-equal KL but different greedy-decode accuracy; minimizing KL
  past a point doesn't buy answers. Report both.
- **Gate dead-zone:** without the high `gate_lr` the gate stays ~0, zeroing the gradient to the
  encoder (grad ∝ `tanh(gate)`). Keep the separate higher-LR param group.
- **`.gitignore` bit us hard:** a bare `data` pattern matched the `src/conceptformer/data/` *package*
  → it was never committed. Patterns for the artifact dir must be **anchored** (`/data`, `/v2/data`).
- **W&B reads:** name-based lookup hits killed `crashed`/`failed` ghosts (empty history). Use run-ids
  or `state="finished"`. `scan_history` can also page oddly — sanity-check row counts.
- **`CUDA_VISIBLE_DEVICES=N` + `--device cuda:N` = crash** (use `--device cuda:0`).
- **`batch 64` OOMs** on 24 GB.
- Source must be ASCII (ruff RUF flags en-dash/×/≈); 100-char lines; keep ruff+ty+pytest green.
- Commit/push **only when asked**; never commit `data/`, checkpoints, or `memory/`.
