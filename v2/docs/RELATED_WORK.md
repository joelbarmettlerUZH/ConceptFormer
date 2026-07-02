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
(training entities: 10k -> 100k -> 300k -> 1M) x (frozen backbone: Qwen3 0.6B -> 1.7B -> 4B -> 8B
-> 14B/32B) — with unseen-entity generalization (PopQA/EntityQuestions) and causal faithfulness as
the response variables. Why this wins:
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
