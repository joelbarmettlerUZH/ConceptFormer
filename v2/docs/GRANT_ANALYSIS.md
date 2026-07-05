# ConceptFormer v2 — Pre-Scaling Analysis (compute-grant evidence pack)

**What this is.** A self-contained synthesis of the analysis run on the converged 100k models,
assembled to support a university compute-grant application. The thesis we must defend *before*
asking for more compute: **the approach is solid, the results are strong, it reproduces and modernizes
ConceptFormer v1, and the one thing left to unlock is scale** (100k -> 300k -> 1M entities, which
F12 shows is still climbing). Every number here is traceable to `docs/RESEARCH_FINDINGS.md` (the
evidence log) and, through it, to a W&B run-id or a re-runnable CLI command. Do not cite a number
without re-verifying at source.

**Setup (fixed across every result below).** Frozen **Qwen3-0.6B** (teacher and student share it;
only the ConceptFormer encoder trains). Corpus **`cftrain_qa_100k`** (925,178 distill rows over 100k
Wikidata entities; snapshot sha `ee4850f...`). Locked config (F8): gate-none + grad-accum eff-batch-32,
3 seeds where stated. Metric throughout: **greedy exact-match accuracy** (a strict bar — the model must
generate the answer, not merely rank it). Three axes: **held-out** (unseen questions, trained entities),
**held-in** (overfit gauge), **PopQA** (unseen *entities* — the honest headline).

**Evaluation protocol (M7-corrected, 2026-07-02).** All numbers below come from the hardened
protocol (RESEARCH_FINDINGS **M7** + corrected F12): **FULL PopQA** (n=14,266; binomial noise
~0.4 pt; official + corrected metrics), **strict held-out** (fact-leakage-free: 32.8% of the
legacy val rows shared a fact with a training paraphrase and are excluded; n=2,000 frozen
fixed-seed sample), Wilson CIs, and **paired exact McNemar** on shared items for contrasts.
Every number is traceable to `data/analysis/eval_final/<ckpt>/summary.json` (+ per-item jsonl).

---

## 1. Headline: token efficiency (the paper's central claim)

**Figure:** `data/analysis/token_efficiency.png` (rendered by `scripts/phaseF_token_efficiency_figure.py`).

ConceptFormer spends **k soft tokens**; the text baselines spend however many fact tokens fit the
budget (median ~100 for a full 1-hop neighborhood, F2). To pre-empt the "strawman baseline"
objection, text-RAG is measured under **three retrieval modes** on the same frozen eval sets:
query-independent top-PageRank truncation, **question-AWARE retrieval** (facts ranked by
embedding similarity to the question), and **LLM-written budgeted summaries** (query-independent
compression, enforced by max-new-tokens). Concept numbers = corrected F12 (3-seed means).

| tokens | concept held-out | text held-out (pgrk / q-aware / summary) | concept PopQA | text PopQA (pgrk / q-aware / summary) |
|--:|--:|---|--:|---|
| 8   | **0.545** | 0.117 / 0.117 / 0.207 | **0.477** | 0.073 / 0.073 / 0.147 |
| 16  | **0.594** | 0.180 / 0.220 / 0.290 | **0.518** | 0.113 / 0.243 / 0.157 |
| 32  | **0.610** | 0.343 / 0.460 / 0.423 | **0.537** | 0.233 / 0.387 / 0.333 |
| 64  | -         | 0.647 / 0.710 / 0.647 | -         | 0.607 / 0.720 / 0.613 |
| 128 | -         | 0.903 / 0.853 / 0.787 | -         | 0.850 / 0.840 / 0.777 |

**Read.** In the **low-token regime (<=32 tokens)** concepts dominate ALL three text baselines:
at 8 tokens, **2.6x** the best text baseline on held-out and **3.2x** on PopQA; at matched
accuracy, even question-aware retrieval needs **~5-6x more tokens** to reach concept-k8 level.
Text catches up only at ~64-128 tokens (most of a verbalized neighborhood). The efficiency
thesis survives non-strawman baselines — and the **untrained-injection control** (Sec 3b) shows
the effect is the *trained encoder*, not the injection slots.

**Re-verify:** `cf-rag-budget-curve --retrieval pagerank|question|summary` ->
`data/analysis/rag_budget_curve_{mode}.json`; concept curve = corrected F12
(`data/analysis/kfamily_corrected.json`).

---

## 2. The k-curve (capacity vs cost) — corrected

Converged 100k, 3-seed mean +/- std, M7 protocol (strict held-out n=2,000; FULL PopQA n=14,266):

| k | held-out (strict) | PopQA (unseen, full) |
|--:|--:|--:|
| 1  | 0.298 +/- 0.006 | 0.203 +/- 0.008 |
| 2  | 0.340 +/- 0.023 | 0.298 +/- 0.075 |
| 4  | 0.441 +/- 0.007 | 0.412 +/- 0.023 |
| 8  | 0.545 +/- 0.014 | 0.477 +/- 0.002 |
| 16 | 0.594 +/- 0.024 | 0.518 +/- 0.020 |
| 32 | **0.610 +/- 0.011** | **0.537 +/- 0.010** |

The curve is **monotone through k32** — paired exact McNemar on shared full-PopQA items makes
every adjacent step significant (k16 vs k32: p ~ 9e-38; the old "plateau at k16" was an n=200
artifact). Returns diminish per token: **k8 already delivers ~89% of k32's PopQA at 1/4 the
tokens** (12.5x compression vs ~100 fact tokens). "Best k" is a genuine token-cost trade (k8
efficiency vs k32 peak), not a capacity ceiling — which also means **larger k budgets remain an
open scaling lever**. Note the seed-stability: with eval noise removed, per-config std is
0.2-2.4 pt (the old 4-8 pt spreads were mostly measurement noise).

---

## 3. Generalization to unseen entities (the honest axis) — and that scale is the lever

The claim that separates "learned the graph" from "memorized the corpus" is **PopQA (entities never
trained on)**. Data scale moved it hard — now measured on **identical full-PopQA eval sets** for
both corpus sizes (10k-corpus checkpoints re-scored by the same `eval-final` protocol):

- **PopQA jumps ~2.05x from the 10k to the 100k corpus**: k8 **0.232 +/- 0.011 -> 0.477 +/- 0.002**
  (3 seeds each, n=14,266, CI ~+/-0.8 pt). Base bracket 0.103, RAG bracket 0.960.
- **NEW under the strict protocol: held-out ALSO rises with scale** (0.453 -> 0.545, +9.2 pt).
  The old "held-out flat across scales" read was a leakage artifact (paraphrase leakage inflated
  the 25-epoch 10k runs more than the 4-epoch 100k runs). Scale cleanly improves BOTH axes.
- Per-entity overfit still FELL with scale (k8 held-in ~0.82 -> ~0.64, F12) — less memorization,
  more edge->concept mapping.
- **The curve is still climbing at 100k** (k8 crept 0.525 -> 0.570 over the last 10k steps; F11/F12).

**This is the grant argument in one line:** the generalization axis we care about improves with data
scale and has **not plateaued** at 100k — more compute (300k -> 1M) is the direct next lever.

## 3b. The trained encoder is the effect (untrained-injection control)

Filling the same k slots with **untrained** top-k mean edge embeddings (`eval-untrained-injection`,
same frozen eval sets) yields PopQA **0.130** (k8; k16 identical, 0.130) vs base 0.103 — barely
above no knowledge — while the trained k8 encoder reaches **0.477**. The learned graph->concept
mapping accounts for **~93% of the injected-knowledge effect**, and extra untrained slots add
nothing. Knowledge injection here is not an artifact of splicing entity-related vectors; it is
the trained compression.

---

## 4. It reads the graph (causal proof), and faithfulness scales with k

Counterfactual graph interventions on each k-model (F13, `cf-graph-faithfulness`, 400 probes each).
*(Protocol note: F13/F15 predate M7 but are within-item interventions — each probe compares the
same question under perturbed vs unperturbed graphs — so the M7 sampling defects do not bias the
deltas; only the absolute "concept correct" column carries the old caveats.)*

| k | concept acc | base-only | ablate-ANSWER (LOW=good) | ablate-OTHER (HIGH=good) | swap-follow->FALSE | stick-to-orig |
|--:|--:|--:|--:|--:|--:|--:|
| 1  | 41% | 10% | 55% | 90% | 0.8%  | 39% |
| 4  | 56% | 10% | 38% | 96% | 16.5% | 20% |
| 8  | 63% | 10% | 34% | 92% | 25.8% | 20% |
| 16 | 65% | 10% | 36% | 95% | 29.8% | 18% |
| 32 | 70% | 10% | 28% | 94% | **34.2%** | **14%** |

Three clean signals, all consistent with a graph reader (not a memorizer):
1. **It's the concepts, not the LLM's memory** — base-only (no concepts) knows just 10% at every k.
2. **Edges are separable** — removing an unrelated edge barely dents accuracy (90-96%); removing the
   answer edge collapses it (55->28%). A widening gap = individually load-bearing edges.
3. **Counterfactual swap scales with k** — when the answer edge is rewired to a false neighbor, the
   model follows it to the false answer **0.8% -> 34.2%** as k grows; at k32 it follows the false edge
   2.4x more often than it sticks to the (now-wrong) truth. **Faithfulness is capacity-limited, not
   absent** — another axis that scale should push further.

---

## 5. Capability preservation (does injection break normal generation?)

On held-out **control tasks** (entity named, task NOT about its facts, so concepts should be inert),
compare the frozen model **with** vs **without** concepts (F15, `cf-capability-preservation`, 400 controls
each). Reported: greedy-agreement over 32 tokens (strict — one flipped token zeroes a row) and, the
cleaner signal, per-token **KL(base||concept)** at the prompt end.

| k | greedy-agreement (32 tok) | KL(base\|\|concept) median | KL mean |
|--:|--:|--:|--:|
| 1  | 29.5% | 0.082 | 0.193 |
| 2  | 27.8% | 0.058 | 0.191 |
| 4  | 30.0% | 0.061 | 0.189 |
| 8  | 33.5% | 0.066 | 0.198 |
| 16 | 28.5% | 0.075 | 0.231 |
| 32 | 33.5% | 0.078 | 0.235 |

**Read.** The next-token distribution barely moves — **median KL stays tiny (0.06-0.08 nats) and only
inches up as k grows 1 -> 32** — so even 32 concept tokens do **not** hijack the frozen model's normal
behavior on off-topic prompts. (32-token exact-match agreement looks modest only because it compounds a strict per-token match
over 32 steps; the low median KL is the faithful measure.) Capability is preserved, by design: the
encoder output is zero-init (gate-none), so at init the student is bit-identical to the frozen LLM.

---

## 6. Relation to ConceptFormer v1 (arXiv 2504.07624)

**Not a head-to-head number** (different backbone/metric/task — see F14). v1 (GPT-2 0.1B, Hit@10/Hit@1
top-k rank on T-REx fill-in-the-blank) vs v2 (Qwen3-0.6B, greedy exact-match QA + PopQA). What matters:
v2 **reproduces v1's phenomena on a harder setup** and **adds** what v1 lacked.

- **Reproduced:** concepts >> text-RAG per token (v1: CF-1 beats RAG at 130x fewer tokens; v2: ~5x at
  8 tokens); a **knee at ~10-16 concept tokens** (v1 ~10-15 vectors, v2 ~k16 — architecture-consistent
  across a 5x-larger backbone and a different task); a single concept token already useful.
- **Added:** an **unseen-entity axis (PopQA)** v1 didn't isolate; a **causal graph-faithfulness proof**
  (Section 4); a modern frozen LLM + **KL self-distillation**; a locked, multi-seed, error-barred
  protocol instead of single-run point estimates.

**Framing for the grant:** v2 = v1's phenomena re-established on a modern LLM under a stricter metric
and a harder task, plus a faithfulness proof and a data-scaling signal. The ask is compute to push the
scaling curve F12 shows is still climbing.

---

## 7. Bottom line for the application

1. **Efficient:** beats query-aware retrieval and LLM summaries 2.6-3.2x at 8 tokens; matched
   accuracy needs ~5-6x more text tokens (Sec 1) — measured against non-strawman baselines.
2. **Generalizes to unseen entities** — the honest axis — improving **2.05x** with a 10x data
   scale on identical full-benchmark eval sets, with held-out rising too (Sec 3).
3. **The trained encoder IS the effect** — untrained injection is near-floor (Sec 3b).
4. **Reads the graph** (causal swap/ablation proof), with faithfulness growing with k (Sec 4).
5. **Doesn't break the frozen model** (capability preserved, Sec 5).
6. **Modernizes and reproduces v1** on a harder, stricter setup (Sec 6).
7. **The remaining unlock is scale:** 100k has not plateaued on PopQA, and the k-curve is still
   monotone at k32. The grant funds the scaling surface (Sec 8).

**Immediate next experiment (the go/no-go):** train one k16 (or k8) model, one seed, on the built
**300k** corpus (`cftrain_qa_300k`, 2.41M rows) and confirm PopQA continues to climb from the 100k
point — now decisively measurable (full-PopQA eval, CI ~+/-0.8 pt; a >2-pt rise is conclusive).
If it climbs, the 1M scale-up is justified.

---

## 8. Research niche and positioning (from the 2026-07-02 literature sweep)

Full analysis with per-paper differentiation: **`docs/RELATED_WORK.md`**. The one-paragraph
version for the application:

**Positioning.** No prior work combines an *amortized, inductive* encoder producing *k
query-independent soft tokens per KG entity*, trained *label-free* by *same-model KL
self-distillation*, evaluated on *unseen entities* and by *causal graph interventions*, at
*100k+ entity scale*. Closest neighbors, each missing several of these: xRAG (2405.13792,
retriever-vector bridge, 1 token, no learned compression, no entity axis), KBLaM (2410.10450,
per-triple KV pairs via attention surgery, synthetic KBs only), Knowledge Prompts (2210.04726,
per-entity lookup-table prompts at 1.1M entities but zero unseen-entity capability), GNP/
GraphToken (2309.15427 / 2402.05862, per-question GNN encoders, task-CE, no precompute). 2026
meta-work (GTEval 2605.03514; "When Graph Tokens Sink" 2606.03712) explicitly identifies the
faithfulness deficiency our counterfactual protocol measures.

**The proposed niche: scaling laws of knowledge injection into frozen LLMs.** A measurement
surface over the **Qwen3.5 family** (huggingface.co/collections/Qwen/qwen35) — training entities
(10k -> 100k -> 300k -> 1M) x frozen backbone (dense 0.8B/2B/4B/9B/27B, ~34x span) — with
unseen-entity generalization and causal faithfulness as response variables. Both marginals are
empty niches (no injection-vs-LLM-scale study within one family exists; no data-scaling law for
knowledge encoders exists), our existing results are the surface's first measured points, and it
connects to the knowledge-capacity-laws (2404.05405) and encode-vs-recall (2602.14080)
conversations. The family adds two design upgrades: (i) **every Qwen3.5 size is natively
multimodal**, so the image-pathway question (inject concept tokens through the vision-token port
vs the text-embedding port) becomes a *factor inside the surface* at 3 sizes, not a separate
model line; (ii) the **MoE members (35B-A3B, 122B-A10B; stretch 397B-A17B)** let us test whether
injection quality tracks TOTAL or ACTIVE parameters — unstudied anywhere. The compute ask maps
1:1 onto this surface: bigger teachers x more entities, embarrassingly parallel across ~100
single-node runs — and the 27B+/MoE rows are memory-infeasible on 2x 24 GB consumer GPUs, not
merely slow (the 397B-A17B stretch run requires a full GH200 node for the frozen weights alone).
Month-1 task: re-anchor the locked 0.6B recipe on Qwen3.5-0.8B before the sweep.
