# Swiss AI Initiative Small Project Grant — DRAFT proposal (2026-07-02)

> Working draft. Format target: research plan <= 2 A4 pages (11pt, 1.5 spacing, 2cm margins),
> excluding cover page, references, appendix. Cover page + CVs + signatures (PI: TBD) not
> included here. Submit via swiss-ai.org/grants; rolling review on the 1st of each month.

---

## COVER PAGE (placeholders)

- **Title:** Scaling Laws of Knowledge Injection into Frozen Language Models
- **Scientific lead:** Joel Barmettler (PhD candidate, University of Zurich)
- **Swiss PI (signs; permanent/tenure-track, >=50% research FTE):** [PROF NAME, UZH]
- **Additional team:** [optional: member with CSCS/Alps or multi-node experience]
- **Requested allocation:** 30,000 GH200 GPU-hours, 6 months (see Appendix A)
- **Open artifacts:** corpora (10k-1M entities), all checkpoints, eval harness, results dataset

---

## RESEARCH PLAN (2 pages)

### 1. State of the art and the gap

Large language models access factual knowledge two ways: stored in weights — uneditable,
expensive to update, and poorly recalled for long-tail entities even when demonstrably encoded
(the encode-vs-recall bottleneck, arXiv 2602.14080) — or retrieved as text, paying hundreds of
context tokens per query. A third channel is emerging: *compress knowledge into a few continuous
"soft tokens" consumed by a frozen LLM*. Text-compression methods (xRAG 2405.13792, ICAE
2307.06945, PISCO 2501.16075) compress retrieved passages; KG-injection methods (GraphToken
2402.05862, GNP 2309.15427, KBLaM 2410.10450) encode graphs per question. Yet the field has no
answer to the two questions that determine whether this channel matters at scale: **how does
injection quality scale with the size of the frozen model, and with the amount of knowledge the
encoder is trained on?** No study measures either — existing work reports 1-2 model sizes
anecdotally, and no data-scaling law for a knowledge encoder exists (survey and per-paper
differentiation: repo `docs/RELATED_WORK.md`). Meanwhile 2026 meta-analyses (GTEval 2605.03514,
"When Graph Tokens Sink" 2606.03712) show current graph-token systems are not faithful carriers
of structure — and call for exactly the causal evaluation methodology we contribute.

**Preliminary results (our own, peer-reviewed + public repo + W&B).** ConceptFormer v1
(Barmettler, Bernstein & Rossetto) was published at **The Web Conference 2026** companion
proceedings (WWW Companion '26, pp. 587-596, DOI 10.1145/3774905.3794653) and received the
**best paper award of its hosting workshop**, establishing on GPT-2 that latent concept
injection beats graph textification by up to +272% Hit@10 at 130x fewer tokens. v2 (this
project) rebuilds the approach on modern backbones: it encodes a Wikidata entity's 1-hop
neighborhood into k soft tokens via a small Perceiver-style resampler, spliced into a frozen
Qwen3-0.6B and trained *label-free* by full-vocab KL self-distillation against the same LLM
reading the facts as text. On two consumer
RTX 4090s we have established, under a hardened protocol (full 14,266-question PopQA, strict
leakage-free held-out, Wilson CIs, paired McNemar, 3 seeds):
(i) **token efficiency** — k=8 soft tokens reach 0.477 on unseen-entity PopQA (frozen base:
0.103), beating question-aware text retrieval and LLM-written summaries 2.6-3.2x at equal token
budget; text needs ~5-6x more tokens for parity;
(ii) **the trained encoder is the effect** — untrained embedding injection scores 0.130,
barely above base;
(iii) **data scaling works and has not saturated** — 10x more training entities lifts
unseen-entity accuracy 2.05x (0.232 -> 0.477) on identical eval sets, with the k-curve still
monotone at k=32;
(iv) **causal faithfulness** — counterfactual edge-swap probes show the model *reads* the
injected graph (swap-follow rises 0.8% -> 34% with k), the intervention-style evidence the
2026 meta-work calls for;
(v) **capability preservation** — median next-token KL 0.06-0.08 nats on off-topic control
tasks: injection does not disturb the frozen model.

**Pilot transect (Qwen3.5, our own hardware).** We additionally ran a two-point model-scale
pilot on the target family (0.8B and 2B, k in {8,16}, 2 seeds/cell, full-benchmark eval): the
unseen-entity injection margin **anti-scales with backbone size at fixed training data**
(+12.9 pt at 0.6B -> +9.4 at 0.8B -> +5.1 at 2B for k8; k16 halves likewise), while the data
axis scales it UP (+12.9 -> +37.4 pt going 10k -> 100k entities). The two axes pull in opposite
directions — whether a larger frozen model simply needs more entities before injection pays,
or is intrinsically harder to steer, is exactly the 2D interaction the proposed surface
measures. The pilot also validated the injection-port factor (O4) and showed the recipe
transfers across model families and attention architectures without retuning.

These are the first measured points of a scaling surface we cannot extend on 24 GB consumer
hardware: every backbone beyond ~4B is memory-infeasible for us, and 1M-entity corpora exceed
our throughput. **The proposal is to complete a de-risked measurement, not to test a new idea.**

### 2. Objectives

Measure and model the scaling behavior of knowledge injection into frozen LLMs on the
**Qwen3.5 family** (dense 0.8B/2B/4B/9B/27B; MoE 35B-A3B/122B-A10B; all natively multimodal;
stretch: 397B-A17B), producing the first empirical scaling laws for this channel:

- **O1 — Model-scale law (dense).** Injection quality (unseen-entity accuracy, faithfulness,
  capability preservation) vs frozen-backbone size, 0.8B -> 27B (~34x), full k-family x 3 seeds
  per size. Key open question either way: do larger frozen models accept soft knowledge better
  (soft-prompt scaling, 2104.08691) or does their parametric knowledge crowd it out?
- **O2 — Data-scale law.** Unseen-entity generalization vs training entities (100k -> 300k ->
  1M Wikidata entities) at 0.8B, with interaction corners at 9B/27B. Analogue, for *injected*
  knowledge, of parametric knowledge-capacity laws (2404.05405).
- **O3 — MoE: total vs active parameters.** Does injection track active (3B/10B/17B) or total
  (35B/122B/397B) parameters? Unstudied anywhere; three natural-experiment points.
- **O4 — Injection port (multimodal).** Same encoder, two ports: text-embedding stream vs the
  vision-token pathway (VLMs are pretrained to consume continuous out-of-vocabulary tokens —
  is that a better landing pad for soft tokens?). Runs *inside* the same family at 3 sizes.
  **Pilot completed on our own hardware (Qwen3.5-0.8B, 2 seeds, paired full-benchmark eval):
  the vision port works untuned and reaches parity with the text port at k=8, but trails at
  k=16 — the interface choice is real and non-trivial.** Whether it flips at sizes with
  stronger vision pretraining is exactly what the funded sweep measures.
- **O5 — Community baselines.** The missing trained baselines evaluated on one harness:
  xRAG-style retriever-vector bridge, per-entity prefix-tuning, on-policy distillation (GKD),
  2-hop rate-distortion ablation.

### 3. Activities and timeline (6 months)

M1: corpus scale-up (1M entities; 300k built), Qwen3.5-0.8B re-anchor + port pilot validation.
M2-M3: O1 dense transect + O3 MoE runs; per-size teacher extraction, HP screens, capacity
re-sweeps. M3-M4: O2 data transect + interaction corners. M4-M5: O4 port factor, O5 baselines.
M5-M6: cross-cutting evaluation (full PopQA, EntityQuestions, faithfulness/capability sweeps),
law fitting, paper + artifact release. All workloads are independent single-node runs
(embarrassingly parallel; high utilization; the 122B/397B runs use single-node multi-GPU
sharding). Evidence of experience: the public v2 repo (dual-GPU orchestration, vLLM serving,
resumable pipelines, 234-test CI) + [TEAM MEMBER]'s Alps/multi-node record [TBD].

### 4. Deliverables and open-science artifacts

(1) **The scaling-law paper** (target: top ML venue) with fitted laws + the causal-faithfulness
methodology. (2) **Open corpora**: 100k/300k/1M-entity Wikidata QA-distillation corpora with
sha-manifested snapshots (CC0 source). (3) **All checkpoints** (~200 encoders across the
surface) on HuggingFace. (4) **The evaluation harness** (frozen eval sets, per-item dumps,
paired statistics — already open). (5) **The results dataset** (every run's per-item outputs),
enabling third-party re-analysis without any GPU.

### 5. Novelty and impact

Novelty: the conjunction — amortized *inductive* per-entity soft tokens, label-free same-model
distillation, unseen-entity + causal evaluation, at scale — exists nowhere (RELATED_WORK.md);
both scaling laws are unmeasured. Impact: (a) for Swiss/European AI — a provenance-carrying,
*editable* knowledge channel for frozen open models (every concept token traces to KG edges;
counterfactual tests prove the model uses them): a hallucination-control and knowledge-
governance primitive, aligned with trustworthy-AI goals; (b) practical — RAG-class knowledge at
~1/10 the context cost benefits every deployment where context is the bottleneck; (c)
scientific — the laws tell the field whether this channel *deserves* scale-up, either way.

### 6. Data and ethics

Wikidata (CC0), PopQA/EntityQuestions (public benchmarks), Gemma-generated questions (Apache-2
model, generation prompts released). No personal data beyond public encyclopedic facts; no
human subjects. Institutional data-management compliance via [UZH group policy — TBD].

---

## APPENDIX A — Technical execution plan and compute justification

### A.1 Cost model (anchored in measured wall-clock, public W&B runs)

- **Anchor A.** One converged training run (Qwen3-0.6B backbone, 100k-entity corpus = 925k
  distill rows, 100k optimizer steps ~ 4 epochs, live teacher, all trajectory evals) measured
  **6.0 wall-hours on one RTX 4090** (W&B group `phaseC-kfamily-100k`, 6 runs / 18h / 2 GPUs).
- **Hardware conversion.** RTX 4090 -> GH200 assumed **2x** realized speedup (conservative:
  bf16 peak ratio is ~6x, but the workload includes generation-heavy eval phases and
  small-batch attention that do not saturate H100-class tensor cores).
- **Convergence margin h = 1.4** — larger k and larger backbones were still improving at the
  fixed horizon (F11/F12); budgeting 40% longer horizons avoids under-trained law points.
- **Unit run cost:** U(P, D) = U0 x P x D, with **U0 = 6.0 / 2 x 1.4 = 4.2 GH200-h**;
  P = active-parameter ratio vs 0.8B; D = entity ratio vs 100k. (Both teacher and student
  forward/backward scale linearly in backbone size; steps scale linearly in corpus size at
  fixed epochs. Activation-checkpointing overhead at >=9B is folded into h.)
- **Corpus prep** (tier + teacher-path extraction; measured 13.5 4090-h per 100k corpus at
  0.6B): prep(P, D) = 6.8 x P x D; for >=9B backbones teacher decoding moves to vLLM
  (measured several-x speedup): x 1/3.
- **Definitive eval suite** per checkpoint (full PopQA 14,266 + strict held-out 2,000 +
  faithfulness/capability probes): E(P) = 0.7 x P.
- **MoE multi-GPU overhead:** 122B-A10B (244 GB bf16, 2-3 GPUs): effective P 12.5 -> 16.3;
  397B-A17B (~800 GB, one 4-8 GPU node): effective P 21.25 -> 34 (parallel efficiency 60%).

### A.2 Work-package budget (GH200-hours; arithmetic shown)

| WP | content | arithmetic | hours |
|---|---|---|--:|
| 0 | 1M-entity corpus gen (vLLM; 17 4090-h/100k x10 x1.4 tail /2) + 1M prep @0.8B (6.8x10) + 300k completion + 0.8B re-anchor | 120+68+20+30 | 240 |
| 1 | Data transect @0.8B: k{4,8,16,32} x 3 seeds x {300k, 1M} | 12 x 4.2 x (3+10) + evals | 670 |
| 2 | Dense transect @100k, sizes {0.8, 2, 4, 9, 27}B: per size 12 runs (k-family x 3 seeds) + prep + 8-trial HP screen + evals | ~69.6P (65.1P for >=9B); sum over P = {1, 2.5, 5, 11.25, 33.75} | 3,520 |
| 3 | Encoder-capacity re-sweep per dense size (3 capacities x 2 seeds) | 25.2P x 53.5 | 1,350 |
| 4 | MoE @100k: 35B-A3B (69.6 x 3.75) + 122B-A10B (65.1 x 16.3) + 397B stretch (2 runs + prep + evals) | 261 + 1,061 + 412 | 1,730 |
| 5 | Interaction corners: 300k x 9B (6 runs) / 300k x 27B (4 runs) / 1M x 9B (k8 x 3) / 1M x 27B (k8 x 1, stretch) + prep | 975 + 2,029 + 1,700 + 2,218 | 6,920 |
| 6 | Injection-port factor @ {0.8, 4, 27}B: 6 vision-port runs per size + 30% port-dev allowance | 6 x 4.2 x 39.75 x 1.3 | 1,300 |
| 7 | Trained baselines: xRAG-style bridge @ {0.8, 9}B, GKD, per-entity prefix-tuning, 2-hop ablation | 216+19+140+139 | 520 |
| 8 | Cross-cutting eval: faithfulness + capability + EntityQuestions + robustness across ~170 checkpoints | ~170 x E(avg P) + integration | 900 |
| | **Core subtotal** | | **17,150** |
| | Contingency 25% (restarts, failed runs, new-cluster debugging; our run history motivates this) | | 4,290 |
| | **Base plan** | | **21,440** |
| | Stretch tier (named): full k-family at MoE + corners (+2,600); 3M-entity point @0.8B (+1,300); ports at 2B + 9B (+470); double-horizon convergence verification @0.8B (+400); law-refinement reruns (+800) | | 5,570 |
| | **Requested** | | **~27,000 -> request 30,000** |

The 3,000-hour margin over the itemized 27,010 covers the single largest uncertainty — the
4090->GH200 conversion factor (a 2x vs 3x assumption moves the whole budget +/-30%). **Fallback
plan at 20k hours:** drop the stretch tier and the 1M x 27B corner (-7,800) — the two core laws
(O1, O2) and the MoE/port factors survive intact.

### A.3 Feasibility evidence

**Track record:** the predecessor system was peer-reviewed and published at WWW Companion '26
(Barmettler, Bernstein & Rossetto, DOI 10.1145/3774905.3794653) and won the **best paper award**
of its hosting workshop — the proposed program extends a line of work this team has already
carried through review once.

Public repo (github.com/joelbarmettlerUZH/ConceptFormer, branch conceptformer-v2): full
pipeline (corpus generation via vLLM, teacher extraction, training, evaluation) is resumable,
cached, and CI-gated (ruff/ty/pytest, 234 tests); all cited numbers trace to W&B run-ids
(entity university-of-zurich, project conceptformer-v2) and to per-item eval dumps. The
workload decomposes into ~200 independent single-node jobs; only 122B/397B require intra-node
sharding. A Qwen3.5-0.8B pilot (text-port re-anchor + vision-port smoke test) runs on our own
hardware before submission, de-risking O4 and the family migration.

## APPENDIX B — References

ConceptFormer v1: Barmettler, Bernstein & Rossetto, "ConceptFormer: Towards Graph-Native
Grounding of Large Language Models via Latent Concept Injection", WWW Companion '26,
pp. 587-596, DOI 10.1145/3774905.3794653 (**best paper award**, hosting workshop; arXiv
2504.07624). Other works — arXiv: 2405.13792 (xRAG); 2410.10450 (KBLaM); 2210.04726
(Knowledge Prompts); 2309.15427 (GNP); 2402.05862 (GraphToken); 2402.07630 (G-Retriever);
2501.16075 (PISCO); 2506.06266 (Cartridges); 2307.06945 (ICAE); 2304.08467 (gist);
2104.08691 (Power of Scale); 2404.05405 (knowledge capacity); 2602.14080 (encode-vs-recall);
2605.03514 (GTEval); 2606.03712 (graph tokens sink); 2509.19371 (knowledge-infusion scaling);
2510.17800 (Glyph); 2512.03643 (optical compression critique). Full annotated survey:
`docs/RELATED_WORK.md`.
