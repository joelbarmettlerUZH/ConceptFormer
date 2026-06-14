# ConceptFormer v2 — Model Architecture & Training Design

Literature-grounded design for the encoder + injection + distillation trainer. Synthesises four
research threads (resampler/soft-prompt injection, KG→LLM injection, positional embeddings for
injected tokens, and the compression-vs-knowledge question). Citations are arXiv ids; ones marked
✓ were verified via `hf papers info`.

> **The one thing to internalise:** under the current objective (distil one fixed verbalization →
> answers derivable from those exact facts), the *globally optimal* encoder is a per-entity
> **autoencoder** — i.e. "just context compression." Genuine graph knowledge does **not** come for
> free; it must be *engineered in* with training pressure and *proven* with held-out probes. See §4.

---

## 0. Setup recap

- Frozen **Qwen3-0.6B** (decoder-only, RoPE, bf16). LLM weights never updated.
- **Encoder** maps one entity's *complete* 1-hop Wikidata neighborhood (variable `N` edges, each a
  `(property_label, neighbor_label)` pair, plus the center label) → `k` constant **concept tokens**
  (`k` ≈ 1–16) at `d_llm`.
- **Student** = frozen Qwen3 with the `k` concept tokens spliced into `inputs_embeds` where the
  verbalized-facts text block would sit; question text kept.
- **Teacher** = same frozen Qwen3 reading the facts as text.
- **Objective** = KL self-distillation: teacher-force the teacher's greedy path, minimise
  per-position `KL(P_teacher ‖ P_student)` over the full vocab. (Token-efficiency accounting for
  `k` vs RAG facts is already wired into the eval — see `eval/tokens.py`.)
- **v1** (ConceptFormer, arXiv:**2504.07624** ✓, Barmettler) = flat cross-attention (center=query,
  neighbor=value, neighbor⊕edge=key), labels featurized by averaged frozen-LLM hidden states, output
  dim coupled to `d_llm`, hard next-token CE. `n≈10–15` concept vectors saturate; even `n=1` beats
  text-RAG at ~130× fewer tokens. **This is the baseline v2 must beat.**

---

## 1. Architecture decision: latent-query resampler (NOT a GNN, NOT an MLP)

**Encoder = a 2–3 layer Perceiver/Flamingo-style latent-query resampler.** `k` learned latent
queries cross-attend over the set of `N` edge feature vectors → exactly `k` outputs, independent of
`N`. This is the Q-Former's useful half (learnable queries + cross-attention bottleneck) without its
contrastive baggage.

Rationale across threads:
- **MLP projector (LLaVA-style)** needs fixed-length input; our input is a variable-size *set*.
  Pooling first destroys per-fact structure. Keep only as an ablation baseline.
- **Full Q-Former (BLIP-2, 2301.12597 ✓)** ships a text tower + ITC/ITM/ITG objectives to bridge an
  *image*↔text modality gap we don't have — our features are already frozen-LLM hidden states.
  Over-engineering.
- **GNN / message passing buys ~nothing on a depth-1 star.** Every neighbor is distance-1 from the
  center and there are no neighbor–neighbor edges, so one message-passing round = one attention step
  (what v1 already does). GraphToken (Perozzi et al. 2024, arXiv:2402.05862 — not on HF index;
  cf. verified "Talk like a Graph" 2310.04560 ✓) finds **no GNN dominates** and that *featurization*
  and *breaking permutation-equivariance* matter far more than message-passing depth.

**Concrete config**
- Latents = `k` (the queries **are** the concept tokens; one query → one output token, BLIP-2 style).
  Sweep `k ∈ {1,2,4,8,16}`. Gist tokens (2304.08467 ✓) warn **more tokens can hurt** (overfitting;
  k=10 hurt LLaMA-7B). Start small; for multi-relation PopQA expect best `k` a small constant **>1**
  (v1's 10–15 is the relevant scale).
- **Internal `d_model` decoupled from `d_llm`** (the explicit v2 goal; fixes v1's coupling). Pick
  `d_model ∈ {512,768}`; project the frozen-LLM edge features *down* into `d_model` on the way in,
  and a **single output FC** maps `d_model → d_llm` (BLIP-2 pattern). Make this output projection
  expressive — in GraphToken most capacity lived in the projection head, not the graph body.
- Resampler block = `cross-attn(latents ← edges) + self-attn(latents) + FFN`. Start at **2 blocks**;
  add depth only if KL plateaus.

**Per-edge featurization (keep v1's inductive, zero-cold-start idea; align to Qwen3)**
- Embed each `(property_label, neighbor_label)` with the **same frozen Qwen3** we inject into (input
  embeddings or a fixed early-layer mean-pool) → one key/value vector per edge. This is **inductive
  by construction** (any new entity/relation with a label works — no lookup-table cold start) and
  pre-aligned to the LLM space, minimising the graph↔text gap GNP needed a domain projector for.
  Strong precedent: BLP inductive entity reps from text + link prediction (2010.03496), NodePiece
  (2106.12144 ✓), ULTRA (2310.04562).
- **Do NOT use transductive KG embeddings (TransE/RotatE/ComplEx) as the LLM-facing node feature** —
  they break on unseen entities and live in an unaligned space. (GNP 2309.15427 ✓ only uses DistMult
  as an *auxiliary loss*, never the feature — the right instinct.)
- **Center entity is explicit conditioning**, not just another set element (v1 makes it the query).
- **Edge fusion:** v1 adds `K = N·W_K + E` (property added into the neighbor key). Prefer
  **concat-then-project or FiLM/gating of neighbor by relation** so a strong property vector can't
  wash out neighbor identity ("occupation = physicist" must keep "physicist" recoverable). Distilling
  against verbalized `(subject, relation, object)` triples keeps student/teacher information units
  matched (KAPING 2306.04136 ✓).

---

## 2. Injection: zero-init tanh gate so student ≡ frozen model at step 0

Adopt **LLaMA-Adapter's zero-init gating** (2303.16199 ✓). Two placements; do (1) first:

1. **Output-side zero gate (simplest):** inject `embed = tanh(g) · C` with per-token learnable
   `g` **initialised to 0**. At init the concept embeddings are zero-norm → the student's
   distribution over the question is *exactly* the frozen model's → early KL gradient flows into `g`
   first, then the resampler. This is the strongest possible regulariser for the **no-harm /
   capability-preservation** goal and makes early training stable.
2. **Attention-score zero gate (LLaMA-Adapter exact form):** separate softmax over the
   concept-key block scaled by per-head/per-layer `tanh(g_l)=0` at init. More surgical; use only if
   output-gating proves too coarse.

Use **per-token gates** (k of them) so concept tokens can switch on at different rates.

---

## 3. Positional embeddings for the concept tokens

**Give the `k` concept tokens a contiguous, unit-step RoPE block in exactly the slot the
verbalized-facts text occupied for the teacher; the question continues monotonically from there.**
If system tokens end at position `p`: concept tokens get `p+1 … p+k`, question starts at `p+k+1`.
This is the M-RoPE / LLaVA default and satisfies "positional coherence" + "preserve textual priors"
(Qwen team, 2510.23095 ✓).

Empirically-confirmed failure modes to avoid (all from 2510.23095 ✓):
- **Large position gap before the block** ("reserving" the facts span) → model *ignores* the injected
  content. ✗
- **Position-id overlap** between concept tokens and generated text → "modality confusion" / endless
  repetition. ✗
- **Resetting/zeroing the text positions** to accommodate concepts → drops *below* vanilla RoPE. ✗
- **All-k-share-one-position** → RoPE rotation identical for every concept token, so the LLM can't
  positionally distinguish them; off-distribution. ✗ (Solve permutation-invariance in the *encoder*,
  not at the LLM interface.)

**Permutation-invariance belongs in the encoder, not the LLM positions.** Neighbors are an unordered
set → build the encoder permutation-invariant (Deep Sets 1703.06114; Set Transformer **PMA** with `k`
learned seeds, 1810.00825 ✓). The `k` *output slots* then have a canonical order, so the contiguous
RoPE block imposes no spurious neighbor ordering. (If you ever fed raw per-neighbor tokens directly,
sequential positions *would* inject spurious order — another reason invariance lives in the encoder.)

**Match the teacher's position layout** so student question tokens land at the same RoPE offsets as
the teacher's — maximises per-position KL transfer.

*Optional ablation (not the default):* compress the concept block into a small position span with
fractional increments (V2PE, 2412.09616) if long generations show the frozen model under-attending
to the concepts (RoPE long-range decay; Vista-LLaMA 2312.08870). Unlikely to bite for short QA.

---

## 4. Compression vs. genuine graph knowledge — the central design problem

**The tension is real and structural.** If the encoder only ever sees one entity's 1-hop facts and
is only asked to reproduce answers derivable from those exact facts, the optimum **is** an
autoencoder — the ICAE / 500xCompressor regime (2307.06945, 2408.03094), which optimise literal
reconstruction. xRAG (2405.13792) is "good RAG compression" in its strongest published form: one
token per document, frozen LLM, precomputed reusable embeddings — but *no* compositional claim. That
is exactly failure mode **(a) mere compression**.

**Why local-only training *can* still yield genuine reusable representations (b):** inductive KG
embeddings (NodePiece 2106.12144 ✓; GraIL 1911.06962; ULTRA 2310.04562) prove a representation
computed *purely from local neighborhood* is reusable and globally consistent on **unseen** entities
— **on one condition:** the model must encode entities as *functions of shared, entity-agnostic
relational structure*, not memorised per-instance content. The frozen LLM is the lever: it already
holds a globally-consistent semantic space. If concept tokens land *on that manifold* and use
relations the way the LLM already represents them, "local training" becomes *projecting each entity
onto a pre-existing global space* — global consistency is **inherited from the frozen LLM**, not
learned from the global graph. Dark-knowledge distillation (Hinton 1503.02531) transfers the
teacher's similarity structure (not just the answer), which is the mechanism — but only on the
teacher's *support*, so a single fixed verbalization degenerates to memorisation (teacher-hacking;
2502.02671, GKD 2306.13649).

**Hard limit to state honestly:** with 1-hop-only encoding you **cannot** recover a 2-hop fact from
one entity's tokens — information-theoretically impossible. Multi-hop must come from **inference-time
composition** of multiple entities' tokens chained by the frozen LLM (Guu path queries 1506.01094;
Query2Box 2002.05969), with expected cascading error. Frame the (b) claim around *composability +
consistent placement*, not "recovering unshown edges from one entity."

### 4a. Design levers that push toward (b), ranked by expected impact

1. **Distil a *distribution* of diverse queries per entity — never a reconstruction loss.** This is
   the single decision separating us from ICAE/500xCompressor. Many question framings / paraphrases /
   relation-targeted queries per entity, KL on all. Diversity forces a *general* entity rep, not a
   text codec. **(Already partly in place: the CF-Train task mix — Gemma-diverse QA + compositional +
   descriptive + control.)**
2. **Neighbor-subsampling distillation.** Encoder sees the full neighborhood, but build the
   *teacher* from a **varying random subset** of neighbors each step and distil against it. Forces the
   concept tokens to be a *stable* entity representation invariant to which facts were verbalized —
   directly attacks the per-instance squeeze. Cheap, high impact. **(New — add to the trainer.)**
3. **Manifold regularisation:** pull concept vectors toward the frozen `inputs_embeds` statistics
   (match mean/cov, or penalise distance to the real-embedding subspace). Makes tokens land on the
   space the LLM already reasons in → inherits global consistency. Low cost, medium-high impact.
   (Already flagged as optional in the v2 goals memo.)
4. **Cross-entity contrastive / relational alignment:** `concept(Paris) − concept(France) ≈
   concept(Berlin) − concept(Germany)` under a relation operator (RotatE/Query2Box geometry;
   relational KD 1904.05068). Most direct way to manufacture a consistent relational space.
5. **Compositional / multi-relation + two-entity training:** queries needing two relations combined,
   and two-entity prompts where both entities' tokens are spliced. The only way to train the
   inference-time composition we'll evaluate. Higher cost; essential if multi-hop is a headline.
6. **Reverse-/mode-seeking KL or on-policy GKD (2306.13649):** with limited `k` the student can't
   cover the full teacher; mode-seeking avoids smearing mass and reduces teacher-hacking.
7. **Descriptive "essence" auxiliary (lowest):** mild abstraction pressure but risks
   reconstruction-flavoured objective — keep small. (Already a CF-Train task family.)

### 4b. Evals/ablations that PROVE (b) not (a)

Governing principle: **find a query whose answer is NOT a substring of the shown facts, that an
autoencoder therefore cannot serve, and show ConceptFormer answers it.**

- **Neighbor-invariance probe (decisive, cheap).** Build tokens from disjoint neighbor subsets `S1`,
  `S2` of the *same* entity. (a) → different codecs; (b) → near-identical tokens that answer each
  other's held-out questions. Metric: cos(token sets) + accuracy of `S1`-tokens on questions whose
  facts were only in `S2`. High → genuine entity rep, **impossible for a pure autoencoder**. Sidesteps
  the "held-out edge impossible with 1-hop" objection (edge is held out *from the verbalization*, not
  the graph). **This + lever 2 are the matched pair: lever 2 trains for (b), this proves it.**
- **Token-swap counterfactual:** splice entity B's tokens into a prompt about A → (b) coherently
  answers about *B* (portable entity handle), (a) garbles.
- **Linear probe** concept tokens → held-out structured attributes (type, a relation's object) not
  used as training queries. High decodability → structured embedding.
- **NN/retrieval geometry** of the static lookup table: (b) → relationally coherent clusters +
  consistent analogy offsets; (a) → geometry reflects surface fact length/overlap.
- **Multi-hop / compositional held-out (headline):** splice A and B tokens (A `located-in` B), ask an
  A→B chain never trained whose answer is in *neither* entity's shown facts. Compare vs teacher
  reading both verbalizations — metric is *composability*, not raw accuracy.
- **Inductive unseen-entity probe:** tokens for entities the encoder never saw → above-chance probe ⇒
  learned a *function of structure* (the definition of (b)).

---

## 5. Training recipe (defaults)

- AdamW, encoder peak LR **1e-4**, cosine decay → 5e-5, **~2k-step linear warmup**. Gate `g` can use a
  slightly higher LR (starts at 0). LLM frozen throughout; encoder in fp32 master weights, frozen LLM
  bf16.
- **Skip BLIP-2 stage-1 contrastive** (features already in LLM space) — train end-to-end on KL. If
  unstable, a short next-token-CE-on-facts warmup before switching to KL; do **not** import ITC/ITM/ITG.
- KL: teacher-force teacher greedy path, full-vocab per-position KL; optional temperature; optional
  small hard-CE answer anchor (ablate). Reverse/GKD optional (lever 6). **Use KV cache** (fixes v1's
  O(N²) rerun-per-token).
- **First milestone = overfit ~16 smoke examples to near-zero KL** end-to-end (featurize → encode →
  gate-inject → student forward → KL vs teacher). Validates the mechanism before any scaled data run.
- `ConceptFormerPredictor` implements the existing `Predictor` protocol → instant eval parity; it
  already slots into the `k`/uncapped token-efficiency accounting.

---

## 6. Concrete deltas vs v1 (what actually changes)

| Aspect | v1 (2504.07624) | v2 |
|---|---|---|
| Encoder | flat cross-attn, `n` independent Q/K/V blocks | 2–3 layer latent-query resampler, `k` shared latents |
| Internal dim | coupled to `d_llm` | **decoupled** `d_model` 512/768 + output FC |
| Edge fusion | additive `N·W_K + E` | concat/FiLM (neighbor not washed out) |
| Permutation | implicit | explicit Set-Transformer PMA invariance |
| Injection | splice, no gate | **zero-init tanh per-token gate** (no-harm at init) |
| Positions | GPT-2 hand-rolled | contiguous unit-step RoPE block in facts slot |
| Objective | hard next-token CE | **full-vocab KL** + neighbor-subsampling + manifold reg |
| Speed | O(N²) rerun per token | KV-cached teacher-forced path |
| Backbone | GPT-2 / Llama-2 | Qwen3-0.6B (chat) |

---

## 7. Pitfalls (consolidated)

- **No reconstruction/autoencoding loss** — that *is* failure mode (a) (ICAE 2307.06945, 500xCompressor
  2408.03094).
- **No single fixed verbalization per entity** — teacher-hacking / memorise-on-support (2502.02671,
  2106.05945). Vary it (lever 2).
- **Don't claim multi-hop from single-entity tokens** — route through inference-time composition.
- **High in-distribution QA accuracy ≠ knowledge** — only held-out-from-verbalization + swap/probe
  tests discriminate (KD-transfers-accuracy-not-representation, 2505.15442).
- **Graph-token "sink" collapse** (2606.03712): injected tokens can become high-saliency outliers the
  LLM attends to but doesn't *use*. KL-against-text-teacher guards (forces functional equivalence);
  verify with token-knockout / causal tracing.
- **Don't over-`k`** (Gist 2304.08467) — sweep up from 1, stop where *held-out PopQA* peaks (not train KL).
- **Don't couple `d_model` to `d_llm`**; don't starve the output projection (GraphToken).
- **Question-agnostic encoder is harder than KAPING/GNP** — it must compress *all* `N` neighbors into
  `k` without knowing the question. High-degree entities (1000+ edges) stress fixed `k` far more than
  the PopQA popular-entity average; `log()` truncation, budget capacity for it.

---

## 8. Build order (unchanged milestones, now spec'd)

1. **Featurizer** — frozen-Qwen3 label embeddings for nodes/edges (cached, inductive).
2. **Resampler encoder** — Set-Transformer/Perceiver, `k` latents, decoupled `d_model`, output FC.
3. **Gated injection** — zero-init tanh per-token gate; contiguous RoPE block in facts slot.
4. **KL trainer** — teacher (frozen+facts, neighbor-subsampled) vs student (frozen+concepts),
   KV-cached; manifold reg; overfit-16 milestone first.
5. **`ConceptFormerPredictor`** + eval parity (+ token accounting already wired).
6. **No-harm eval** (control/general with vs without injection) + the §4b (b)-vs-(a) probe suite.

---

### Appendix — verified citation ids
✓ 2504.07624 ConceptFormer v1 (Barmettler) · 2301.12597 BLIP-2 · 2303.16199 LLaMA-Adapter ·
2304.08467 Gist Tokens · 2309.15427 GNP · 2510.23095 multimodal positional (Qwen) ·
2106.12144 NodePiece · 1810.00825 Set Transformer · 2310.04560 Talk-like-a-Graph.
Unverified-but-real (not on HF index): 2402.05862 GraphToken (Perozzi et al.), 2204.14198 Flamingo,
2405.13792 xRAG, 2307.06945 ICAE, 2408.03094 500xCompressor, 2310.04562 ULTRA, 1911.06962 GraIL,
2002.05969 Query2Box, 1506.01094 Guu path queries, 1503.02531 Hinton KD, 2306.13649 GKD,
2502.02671 teacher-hacking, 2010.03496 BLP, 2306.04136 KAPING, 2412.09616 V2PE, 2312.08870 Vista-LLaMA,
1703.06114 Deep Sets, 2606.03712 graph-token sinks, 1904.05068 relational KD, 2505.15442 KD-not-rep.
