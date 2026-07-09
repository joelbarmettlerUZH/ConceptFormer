# ConceptFormer v2 — Related Work & Research Niche (2026-07-02)

**Purpose.** The positioning document for the paper's related-work section and the compute-grant
narrative. Compiled from a structured literature sweep (`hf papers search/read` + web verification,
2026-07-02) across three axes: (A) soft-token context compression, (B) knowledge-graph injection
into LLMs, (C) the candidate niche directions. Every claim below carries an arXiv id — re-verify
with `hf papers read <id>` before citing in the paper. Deltas are stated against the v2 design:
Perceiver-style latent-query resampler over an entity's 1-hop (property, neighbor) label-embedding
set -> k soft tokens spliced into a frozen Qwen3-0.6B, trained encoder-only by full-vocab KL
self-distillation against the same frozen LLM reading the facts as text; query-independent
per-entity precompute; unseen-entity (PopQA) + counterfactual edge-swap evals.

**The intersection v2 occupies (no prior work sits in it):** an *amortized, inductive* encoder
producing *k query-independent soft tokens per KG entity*, trained *without labels* by
*same-model KL self-distillation*, evaluated on *unseen entities* and by *causal graph
interventions*, at *100k-entity scale*. Each neighboring paper has some of these properties;
none has the conjunction (see gap table, section 4).

---

## 1. Axis A — soft-token / context compression

| work | id | date | mechanism (one line) | decoder frozen? | ratio |
|---|---|---|---|---|---|
| Gist tokens | 2304.08467 | 2023-04 | attention-mask trick, LM compresses its own prompt | no (full FT) | ~26x |
| AutoCompressor | 2305.14788 | 2023-05 | recursive summary vectors as soft prompts | no (full FT) | segments->50 |
| ICAE | 2307.06945 | 2023-07 | LoRA-LLM encoder -> memory slots, reconstruction | yes | ~4x |
| 500xCompressor | 2408.03094 | 2024-08 | LoRA encoder -> KV values of 1-16 tokens | yes | 6-480x |
| xRAG | 2405.13792 | 2024-05 | frozen retriever doc-vector -> 1 soft token via MLP | yes | ~100-180x |
| LLMLingua-2 | 2403.12968 | 2024-03 | extractive token classifier (HARD compression) | untouched | 2-5x |
| UltraGist | 2405.16635 | 2024-05 | segment-wise gisting, dynamic ratios | no (LoRA) | 2-32x |
| PISCO | 2501.16075 | 2025-01 | memory tokens + sequence-level self-distillation | no (LoRA) | 2-128x |
| CompLLM | 2509.19228 | 2025-09 | per-segment "Concept Embeddings", activation distill | yes (LoRA detach) | 2x |
| ARC-Encoder | 2510.20535 | 2025-10 | 3B encoder, query pooling -> decoder embeddings | yes | 4-8x |
| Cartridges | 2506.06266 | 2025-06 | per-corpus KV cache trained by self-study distill | yes | 38.6x KV |
| Glyph | 2510.17800 | 2025-10 | render text as images, VLM reads pixels | no (CPT+SFT) | 3-4x |

2026 successors worth tracking: ComprExIT (2602.03784, explicit information transmission over
frozen hidden states), density-aware semi-dynamic ratios (2603.25926 — relevant to making k
neighborhood-size-adaptive), ArcAligner (2602.12235's sibling; decoder-side gating for compressed
context, notes long-tail/multi-hop gains), and **token-overflow detection (2602.12235)** — the
axis's only faithfulness-adjacent work, and it is *correlational probing* (0.72 AUC detecting when
an xRAG-style vector lost the answer), not causal intervention.

**Closest single paper to v2: xRAG (2405.13792).** Shares the frozen decoder, tiny trainable
bridge, input-embedding soft tokens, and token-level KL self-distillation against the same LLM
reading the raw text. A reviewer WILL say "xRAG for knowledge graphs". The rebuttal, in order of
strength: (1) xRAG never *learns* compression — it translates a single pre-existing frozen
retriever vector; v2 learns the set->k compression function itself. (2) Text passage vs
relation-typed edge set: the counterfactual edge-swap test is only well-posed because v2's input
factorizes into discrete facts. (3) 1 fixed token (with documented overflow failures — 2602.12235;
ARC-Encoder measures an xRAG-like setup at 26.1 avg EM vs 49.2 open-book) vs a tunable k with a
measured token-efficiency knee. (4) Retriever-in-the-loop per query vs entity-indexed, retrieval-
free precompute with an unseen-entity axis no compression paper evaluates.

**PISCO (2501.16075) rebuttal hook:** PISCO's Appendix G claims frozen decoders cannot reach
parity with decoder-LoRA under compression distillation. v2's frozen-decoder full-vocab-KL result
at entity granularity is a direct counterexample worth one explicit sentence in the paper.

**Cartridges (2506.06266) contrast:** same objective *family* (distill same-model-with-context
into a compact artifact) but per-corpus gradient descent — minutes-to-hours of GPU per item and,
by construction, zero generalization to unseen items. v2 = "amortized Cartridges for entities":
one encoder forward pass, and the encoder *generalizes* (PopQA).

## 2. Axis B — knowledge-graph / structured injection

| work | id | date | granularity | injected where | objective | LLM frozen? |
|---|---|---|---|---|---|---|
| Knowledge Prompts | 2210.04726 | 2022-10 | 1 vector / entity (lookup) | input embeds | triple-completion CE | yes (T5) |
| KAPING | 2306.04136 | 2023-06 | triples as TEXT | prompt text | none (zero-shot) | yes |
| GNP | 2309.15427 | 2023-09 | 1 pooled vector / question | input embeds | answer CE + DistMult | yes / LoRA |
| GraphToken | 2402.05862 | 2024-02 | few tokens / question graph | input embeds | answer CE | yes |
| G-Retriever | 2402.07630 | 2024-02 | 1 token + ~600 text tokens | input embeds | answer CE | yes / LoRA |
| LLaGA | 2402.08170 | 2024-02 | 1 token / node | input embeds | task CE | yes |
| InfuserKI | 2402.11441 | 2024-02 | adapters (weights) | adapter layers | task CE | no |
| KBLaM | 2410.10450 | 2024-10 | 1 KV pair / TRIPLE / layer | every attn layer | synthetic instr. CE | yes |
| FtG | 2412.09094 | 2024-12 | 1 token / query (KGC) | input embeds | answer CE | no (LoRA) |
| NT-LLM | 2410.10743 | 2024-10 | positional node tokens | input embeds | task CE | yes |

2025-26 meta-work that *motivates* v2's evaluation style: **GTEval (2605.03514)** finds
graph-token LLMs "do not fully understand graph tokens" and lean on accompanying text;
**"When Graph Tokens Sink" (2606.03712)** shows graph tokens become attention-sink outliers whose
saliency decouples from graph semantics, via post-hoc pruning/swap interventions. Both papers
diagnose exactly the deficiency v2's *built-in* counterfactual edge-swap protocol measures (and
shows scaling with k). **Knowledge Infusion Scaling Law (2509.19371):** weight-side infusion has a
model-size-dependent memory-collapse point — ammunition for the context-side/frozen-LLM approach.

**The three "how is this different from X" challenges and answers:**
1. **KBLaM (2410.10450)** — precomputed continuous knowledge tokens for a frozen LLM, ICLR'25.
   Differentiate: per-triple KV pairs at every layer (attention surgery, rectangular attention)
   vs k input-embedding tokens per entity *neighborhood* (no model modification); synthetic
   instruction CE — with self-admitted loss of exact strings — vs full-vocab KL that preserves
   the LLM's own text-reading distribution; synthetic/Enron KBs + BERTScore vs Wikidata + PopQA
   unseen entities + causal faithfulness.
2. **Knowledge Prompts (2210.04726)** — per-entity soft prompts, 1.1M Wikidata entities, in 2022.
   Differentiate: lookup-table free parameters (every new entity requires gradient training;
   zero unseen-entity capability) vs an amortized inductive encoder (unseen entity = one forward
   pass — the PopQA axis); 1 unstructured vector vs k tokens from a structured edge set; T5
   triple-completion vs decoder-LLM logit distillation; no efficiency/faithfulness measurement.
3. **GraphToken (2402.05862) / GNP (2309.15427)** — GNN soft tokens into frozen LLMs since 2023.
   Differentiate: per-QUESTION encoders (retrieval + GNN in the inference path) trained with
   task CE on classification-style QA vs query-independent cacheable per-entity tokens trained
   with NO labeled QA (self-distillation) and evaluated causally. G-Retriever's own hallucination
   analysis (single graph embedding cannot carry full structure) argues *for* v2's k>1 latent
   queries.
   **CORRECTION (2026-07-09, source-verified via hf papers; paper S2 fixed accordingly):**
   "per-QUESTION" is only true for GNP + G-Retriever. GraphToken's encoder never reads the
   question (synthetic graphs, no entities, factual grounding named as future work). LLaGA
   encodes NODES question-independently (inductive text-encoder features, zero-shot to unseen
   graphs) but rebuilds sequences per instance and targets node classification / link
   prediction. The defensible scoped claim: none computes entity tokens ONCE and reuses them
   across arbitrary prompts, and none trains without task labels. Also verified: KBLaM evals
   on synthetic + real Enron-derived KB (not "synthetic only") and handles unseen triples via
   its encoder; PISCO = sequence-level distillation + LoRA both sides (our objective is
   token-level KL in PISCO's taxonomy — never say "we share PISCO's sequence distillation");
   xRAG training uses paraphrase pretraining + ~1M labeled instruction examples on top of its
   KL term.

## 3. Axis C — adjacent lineages (context for the grant)

- **Entity-memory architectures** — Entities as Experts (2004.07202), Facts as Experts
  (2007.00849), Mention Memory (2110.06176), QA-Memory (2204.04581): the 2020-22 wave of
  million-entity memories, all requiring trained (non-frozen) LMs; the line died out and has no
  2025-26 successor for modern frozen LLMs. v2 is effectively its frozen-LLM revival.
- **Knowledge-capacity laws** — Physics of LMs 3.3 (2404.05405): ~2 bits/param for *parametric*
  storage; Memory Layers at Scale (2412.09764). No analogue exists for *injected* (context-side)
  knowledge — open measurement space.
- **Encode-vs-recall bottleneck** — "Empty Shelves or Lost Keys?" (2602.14080): frontier models
  *encode* 95-98% of long-tail facts but *recall* only 25-33%. Frames concept tokens as a trained
  recall mechanism rather than a knowledge store — a strong grant framing.
- **Optical/text-as-image compression** — Glyph (2510.17800), DeepSeek-OCR (2510.18234), Text or
  Pixels (2510.18279); contested by "Optical Context Compression Is Just (Bad) Autoencoding"
  (2512.03643: mean-pooling matches the vision encoder at matched ratios). Nobody injects
  *structured* data through a VLM's visual pathway.

## 4. Gap table — what exists vs what v2 does

| property | closest holder | does v2 add something? |
|---|---|---|
| compress STRUCTURED graph (edge set), not text | none in axis A; KBLaM is per-triple KV | yes — set->k resampler, permutation-invariant |
| per-entity, query-independent precompute at scale | Knowledge Prompts (1.1M, lookup table) | yes — amortized encoder, not free parameters |
| inductive: works on UNSEEN entities, measured | none (no compression or KG-injection paper) | yes — PopQA axis, scales with data (F12) |
| same-model full-vocab KL self-distillation | xRAG (partial), Cartridges (per-corpus) | yes — label-free, amortized, entity-level |
| token-cost vs accuracy frontier vs text-RAG | G-Retriever (retrieval savings only) | yes — the token-efficiency figure |
| causal faithfulness of injected tokens | 2602.12235 (probing), 2606.03712 (post-hoc) | yes — built-in counterfactual edge-swap, scales with k |
| injection quality vs frozen-LLM scale (one family) | GNP/G-Retriever/xRAG (2 sizes, anecdotal) | OPEN — the grant ask |
| data-scaling law for the injector/compressor | none | partially — 10k->100k measured, 300k->1M is the grant ask |

## 5. Niche decision (for the grant)

**Lead niche: "Scaling laws of knowledge injection into frozen LLMs."** A 2D measurement surface —
(training entities: 10k -> 100k -> 300k -> 1M) x (frozen backbone: the **Qwen3.5 family** — dense
0.8B/2B/4B/9B/27B, MoE 35B-A3B/122B-A10B, stretch 397B-A17B; all natively multimodal, so the
image-pathway ablation runs inside the same family) — with unseen-entity generalization
(PopQA/EntityQuestions) and causal faithfulness as the response variables. NOTE: all existing v2
evidence is on Qwen3-0.6B; the sweep re-anchors on Qwen3.5-0.8B first. Why this wins:
- Both marginals are EMPTY niches (sections 3-4): no injection-quality-vs-LLM-scale study within
  one family exists, and no data-scaling law for knowledge encoders exists.
- v2's existing results are the first measured points (10k->100k PopQA jump with held-in
  *falling*; faithfulness scaling with k) — the grant funds completing a de-risked measurement,
  not a speculative idea.
- It plugs into two live 2026 conversations: knowledge-capacity laws (2404.05405) and the
  encode-vs-recall bottleneck (2602.14080), while occupying an empty cell of that literature.
- Compute maps 1:1 to the ask (bigger teachers x more entities, embarrassingly parallel), and the
  4090-scale evidence makes the "we are compute-bound, not idea-bound" case self-evident.

**Secondary aim (bounded, high-risk/high-novelty): image-pathway injection.** Nobody injects
structured knowledge through a VLM's visual-token interface. The honest framing is NOT
"image tokens are a higher-bandwidth channel" (contested — 2512.03643) but: *VLMs are pretrained
to consume continuous, out-of-vocabulary token streams; is a vision-pretrained interface a better
landing pad for soft concept tokens than a text-only embedding stream?* Cheap crisp ablation
(same encoder, two injection ports on Qwen3-VL); a positive result is novel, a negative one is a
useful finding about where soft tokens should enter multimodal models.

**Demoted: 2-hop neighborhoods as a standalone contribution.** "Multi-hop subgraph -> GNN -> soft
prompt for a frozen LLM" is occupied since 2023 (GNP) and already has its own evaluation
(2605.03514) and mechanistic-criticism (2606.03712) literature. Keep 2-hop as an *internal
rate-distortion ablation* inside the scaling story (how many facts fit in k tokens as the
neighborhood grows ~100x), where v2's constant-k compression and faithfulness measurement are the
novel parts.

**Baseline obligations the surveys impose for the paper:** compare against (i) an xRAG-style
retriever-vector bridge (trained arm — needs compute, part of the grant), (ii) the untrained
top-k mean-edge-embedding injection (implemented, `eval-untrained-injection`), (iii) query-aware
text retrieval + budgeted LLM summaries at matched token budgets (implemented,
`cf-rag-budget-curve --retrieval question|summary`), and cite-and-differentiate KBLaM + Knowledge
Prompts + GNP/GraphToken in related work (section 2 wording).

## 6. Sweep 2026-07-09 (hf papers, two parallel agents; dedup vs sections 1-3)

**Added to paper (18 new cites):** ftvsrag 2403.01432, icformer 2406.13618, userllm 2402.13598,
persoma 2408.00960, promptdistill 2412.14964 (CLOSEST objective: same-model teacher-reads-text
KL, but into LoRA weights, needs generated QA), disc 2602.16093, kvdistill 2503.10337,
gistsurvey 2412.17483 ("lost if surprise" = our swap-probe failure mode), lgpt 2501.17549,
srki 2511.06446 (KBLaM successor, reusable per-triple latents, supervised attention -> must
defuse), realm 2510.09711 (per-entity discrete tokens, trained vocab, KGC task), gfmrag
2502.01113 (zero-shot cross-graph but retriever-level, LLM reads text), uniglm 2605.12197,
lostinspace 2404.13594 (probing resampler tokens, correlational only), breakingchain
2603.16475, petrov 2310.19698 (soft prompts elicit, cannot teach -> steerability theory),
scalingft 2402.17193, memorizeretrieve 2604.00715 (RAG utility is scale-dependent).

**Doc-only / grant-relevant:** CAG 2412.15605, E2P 2505.17051, DAST 2502.11493, C3 2511.15244,
Gist-COCO 2402.16058, SelfCP 2405.17052, GistPool 2504.08934, K-ON 2502.06257, SpreadsheetLLM
2407.09025, steering reliability 2504.04635, memory-token capacity vs scale 2506.15001, soft
task-embedding injection 2507.20906, steerability testbed 2606.11599, knowledge-injection
survey 2502.10708, GTSQA 2511.04473 (candidate SECOND transfer benchmark: Wikidata-derived,
unseen structures), G-reasoner 2509.24276, BYOKG-RAG 2507.04127, GraphRAG-Bench 2506.05690,
CoLoTa 2504.14462 (long-tail successor benchmark), KG-ICL 2410.12288 + ULTRA 2310.04562
(inductive-KG lineage), ARC-JSD 2505.16415, attention-vs-graph 2505.02130, knowledge homophily
2505.19286, OPCD 2602.12275, GCD 2411.15927, data-centric compression 2602.01778; multimodal
grant cluster: EDT-Former 2602.02742, Mario 2603.05181, KORE 2510.19316, multimodal GFM
2602.04116, PEFT-scaling 2606.02437.

**Gaps confirmed by both agents (claimable):** no counterfactual interventions on injected
graph/soft tokens anywhere; no zero-shot cross-graph transfer of a TOKEN-level KG-LLM
interface; no label-free graph-to-token training; no report of injected-margin anti-scaling
with frozen-model size (frame via petrov + scalingft + memorizeretrieve).
