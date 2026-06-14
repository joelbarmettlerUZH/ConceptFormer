# ConceptFormer v2 — Design Doc / Paper-Delta

Status: **draft for alignment** (no code yet). This is the next iteration of ConceptFormer (master thesis → second paper), not a from-scratch approach. It keeps the core thesis and dataset lineage and rebuilds the machine on a modern, fast, model-agnostic stack with a more stable training objective.

See `memory/conceptformer-v1-architecture.md` for how v1 works and `memory/conceptformer-v2-goals.md` for the goal list this doc commits to.

---

## 1. The thesis (unchanged)

Map an entity's **1-hop Wikidata subgraph** into a small number `k` of **soft "concept" tokens** that are spliced into a **frozen** LLM's *input-embedding* sequence. The LLM then answers as if the triples were in its context — but without spending context-window tokens on verbalized text and without retraining the LLM. The headline scientific knob remains **`k` (concept tokens per entity) vs. accuracy vs. tokens-saved-vs-RAG**.

## 2. What changes in v2 (the delta), at a glance

| Area | v1 | v2 |
|---|---|---|
| Encoder | central-node-as-query, `d` independent single-head attentions | **Q-Former / Perceiver resampler**: `k` learnable latent queries cross-attend the neighbor⊕relation set |
| Internal dim | hard-tied to LLM embed dim | **decoupled** `d_model`, learned projection `d_model → d_llm` |
| Objective | hard CE forcing exact object token at exact position | **KL-distillation from a graph-in-context teacher** (+ optional aux terms) |
| Backbone | GPT-2 / Llama-2, hand-rolled wrappers | any HF causal LM via `inputs_embeds`; default **Qwen3-0.6B (chat)** |
| Prompt format | declarative sentence, char-boundary splicing | **chat templates**, concept tokens spliced at a marked slot |
| Training loop | O(N²) re-run per target token, fp32, DataParallel | single teacher-forced pass, bf16, accelerate/DDP, `torch.compile` |
| Node/edge features | pooled LLM "late" embedding of bare label; PBG vestigial | richer verbalized encodings; **PBG removed** (decision below) |
| Data pipeline | live SPARQL + AWS PageRank + fuzzy boundaries + TAR builders | **frozen, versioned, hashed snapshots**; pydantic schemas; marker-based alignment |
| Config / stack | dataclass + `__post_init__` paths, `requirements.txt`, 18 numbered scripts | **uv + pydantic-settings**, package + CLI |
| Eval logging | W&B hard dependency | pluggable, offline-capable |

## 3. Architecture

### 3.1 Concept encoder (Q-Former resampler) — DECIDED
- Inputs: the subgraph as a **set of edge tokens**. Each edge `(relation, neighbor)` is fused (not "relation added into keys only" as in v1) into one token in `R^{d_model}` — e.g. project `concat(neighbor_repr, relation_repr)`, plus an optional rank/positional feature so high-PageRank neighbors are distinguishable. The central node is included as an additional token.
- `k` **learnable latent query vectors** (`k` = `num_pseudo_words`) cross-attend over the edge-token set through a few transformer blocks (RMSNorm, SDPA/FlashAttention). Variable neighbor counts handled by **padding + attention mask** (removes v1's `DynamicNeighbourBatchSampler` neighbor-count bucketing).
- Output: `k` vectors in `R^{d_model}` → **learned projection** to `R^{d_llm}` → spliced into the backbone's input-embedding stream.
- **Decoupled dim** is a first-class property: `d_model` (e.g. 512) is independent of the backbone; only the final projection is backbone-specific, enabling backbone swaps and (future) one encoder → many backbones.

### 3.2 Backbone adapter
- One thin adapter over `AutoModelForCausalLM`: input embeddings via `model.get_input_embeddings()`, forward via `inputs_embeds` + explicit attention mask, generation via `model.generate(inputs_embeds=...)`. **HF handles positional encoding** for both absolute (GPT-2) and RoPE (Qwen/Llama) — no manual `wpe`/`wte`/BOS/space-token code.
- **Chat-first**: build prompts with `tokenizer.apply_chat_template`; concept tokens replace a sentinel slot inside the user turn; supervision is on the assistant turn.
- Default backbone **Qwen3-0.6B**; larger Qwen/Llama/Mistral/Gemma are config swaps.

### 3.3 Removed / dead code retired
PyTorch-BigGraph (off in v1's final configs), `NodeEdgeEmbedder`/`GraphTransformerNet` alt paths, the `pass`-body `Embedder`, the broken `helpers.py` MSE eval, and the in-RAM 5M-entry embedding cache are all dropped.

## 4. Training objective (KL-distillation) — DECIDED

Primary loss = **KL divergence between two next-token distributions of the same frozen backbone**:
- **Teacher**: backbone with the subgraph **verbalized as text in context** (v1's "text injection" / graph-RAG path), running the answer span.
- **Student**: backbone with the `k` concept tokens in place of that verbalized text.
- Minimize `KL(teacher ‖ student)` over the full answer continuation (teacher-forced), not a single position. This directly optimizes "concept tokens ≈ having the triples in context," is tolerant of synonyms/paraphrase, and removes the "exact token at exact position" brittleness.

Auxiliary / ablation terms (switchable, off by default):
- Hidden-state matching at the answer span (cosine/MSE) for a smoother signal.
- Standard teacher-forced LM CE over the answer span (alias-aware) — kept as the **baseline objective** for the ablation table.

Stabilizers: label smoothing, LR warmup + cosine decay, grad clipping, optional projection-only warmup. Teacher distributions/hidden-states **cached per example** so distillation doesn't recompute the teacher every epoch.

## 5. Training efficiency
1. **Single teacher-forced forward pass** per example (concept tokens + full answer), reading all answer-position logits at once — replaces v1's O(N²) per-token re-run (the dominant slowdown).
2. **Precompute node/relation reprs to disk** (safetensors/memmap), keyed by id — no giant RAM cache.
3. bf16, `accelerate`/DDP, gradient accumulation, `torch.compile`, SDPA/FlashAttention, multi-worker loaders; frozen backbone in inference mode with cached embedding matrix.
4. Cache & batch the teacher pass.

## 6. Data
- **Freeze & version** TRExStar subgraphs + a pinned PageRank snapshot into content-hashed Parquet/HF-Hub artifacts; **no live Wikidata/AWS at train/eval time**.
- **Pydantic schemas** for entity/relation/edge/subgraph/example rows (validated, typed).
- **Marker-based / token-level entity alignment** at generation time → exact, reproducible boundaries (replaces `fuzzywuzzy` + `string.index()` hacks).
- Synthetic sentence generation (TriREx successor) via a modern small instruct model with structured output + validation + recorded prompt/model/version.
- WebQSP: keep **multi-answer (`k`) / multi-token** answers instead of dropping them.
- Lineage and names (TREx → TRExStar → TriREx/TRExBite → WebQSP) preserved.

## 7. Evaluation
- Headline metric stays **top-k hits**, but computed on **generated, alias-aware** answers (hits@k / EM / F1), not the first divergent token.
- One-command head-to-head: **concept tokens vs. graph-in-context RAG vs. no-knowledge**, same code path.
- Efficiency curves: accuracy vs. `k`, and **tokens-saved-vs-RAG** at equal accuracy (the core selling point).
- At least one **fresh benchmark** (e.g. Mintaka or a held-out Wikidata QA split) to avoid TREx-circularity.
- Logging pluggable; runs offline and deterministically.

## 8. Stack & engineering
- **uv + pyproject.toml** (pinned), Python 3.11/3.12.
- **pydantic / pydantic-settings** for all config (replaces dataclass `__post_init__` path soup + nested config-list banks); YAML/CLI driven via `typer`.
- Package layout `v2/conceptformer/...` (encoder, adapter, data, train, eval, cli) instead of 18 numbered scripts with hardcoded `__main__` GPU lists.
- `ruff` + type checking + `pytest`; structured logging; HF/accelerate checkpointing.
- **Static lookup-table deployment kept** and modernized (safetensors, sharded); "encode any Wikidata entity on demand" as a clean API.

## 9. What is deliberately kept (iteration, not rewrite)
Frozen-LLM + injected soft concept tokens; the `num_pseudo_words` sweep as the headline; the TREx/TRExStar/TriREx/TRExBite/WebQSP lineage; attention-based neighbor aggregation (Q-Former is its evolution); top-k eval + RAG baseline; static lookup-table deployment.

## 10. Open questions / risks
- Teacher quality: a 0.6B backbone as its own teacher may give a weak distillation target — may want a larger teacher backbone while keeping the small student, or verify the RAG teacher actually answers correctly before distilling from it.
- Encoder depth/heads, `d_model`, and number of cross-attn blocks — to sweep.
- Whether to keep any global/structural alignment signal at all now that PBG is removed (current call: start without it; add back only if ablations demand).
- Chat-template injection slot mechanics (sentinel token vs. embedding-offset splice) across tokenizers.

## 11. Phased plan (post-alignment)
1. Repo restructure: `git mv` v1 → `v1/`, scaffold `v2/` (uv + pydantic + package + CLI).
2. Data slice: freeze TRExStarLite + TriRExLite into versioned pydantic-schema artifacts.
3. Thinnest end-to-end slice on the Lite data: Q-Former encoder + Qwen3-0.6B adapter + KL-distillation, validating the objective before scaling.
4. Baselines + eval harness (concept vs. RAG vs. none), `k`-sweep.
5. Scale to full data; add fresh benchmark; ablations (KL vs. LM CE, with/without aux terms, `d_model`, `k`).
