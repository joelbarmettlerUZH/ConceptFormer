# ConceptFormer v2 — A Complete Design Breakdown

*Written for a graduate reader. Assumes familiarity with transformers, attention, softmax
cross-entropy, and KL divergence; introduces the rest. Verified arXiv ids in §10.*

---

## 1. The problem

A frozen language model `M` (here Qwen3-0.6B) knows a lot, but it is weak on **long-tail factual
knowledge** — facts about entities that were rare in pretraining (PopQA: the 0.6B model answers
~12% from parameters alone). Retrieval-augmented generation (RAG) fixes this by putting the facts
into the prompt as **text**: verbalize the entity's Wikidata neighborhood and prepend it. That lifts
the same frozen 0.6B model to ~94% on PopQA. The cost is **tokens**: a neighborhood can be hundreds
of tokens, paid on every query, and bounded by the context window.

**ConceptFormer's thesis:** inject the same knowledge as a *small constant number `k` of continuous
"concept tokens"* spliced into the model's input-embedding sequence, instead of hundreds of text
tokens. If `k` concept tokens can recover most of the RAG accuracy, you get the knowledge at a
fraction of the token cost, the concept tokens can be **precomputed once per entity** (a static
lookup table), and you are no longer context-window-bounded.

Formally. Let an entity `e` have a 1-hop neighborhood `G_e = {(r_i, o_i)}` (relation, object pairs).
- **Teacher (RAG):** condition `M` on `x_T = [system, verbalize(G_e), question]`.
- **Student (ConceptFormer):** condition `M` on `x_S = [system, C_e, question]`, where
  `C_e = Encoder(G_e) ∈ ℝ^{k×d}` are concept-token embeddings spliced where the facts text sat.

We train `Encoder` so the student reproduces the teacher's output distribution. `M` is **never
updated**. This document explains every design choice, why it is sound, how it differs from v1, and
where it can fail.

---

## 2. Notation

- `d = d_llm` — the LLM input-embedding dimension (1024 for Qwen3-0.6B).
- `N` — number of 1-hop edges of the current entity (variable).
- `k` — number of concept tokens (constant hyperparameter, ~1–16).
- `V` — vocabulary size. `M(· | x) ∈ Δ^{V}` — the next-token distribution given prefix `x`.
- `P_T`, `P_S` — teacher / student per-position distributions.
- `KL(P‖Q) = Σ_v P(v) log P(v)/Q(v)` — relative entropy.

---

## 3. The v1 baseline (what we are improving on)

ConceptFormer v1 (arXiv:2504.07624, the published thesis) established the core idea and that it
works. Its pipeline:

- **Features.** Node/edge vectors = pooled **last-hidden-state** embeddings of the label text from
  the frozen LLM (a `BigGraphAligner`; the PyTorch-BigGraph alignment was vestigial/off).
- **Encoder.** A `GraphAttentionEmbedder`: the center node is the query, `K = key(neighbor) + edge`
  (the relation **added** into the key), `V = value(neighbor)`. It runs `n` *independent single-head
  attentions* to emit `n` concept vectors. Output dim **coupled** to `d_llm`.
- **Injection + objective.** Build `[start-text emb][concept vecs][end-text emb]`; **hard
  cross-entropy** on the object tokens. The forward pass rebuilt a padded sequence per target token
  and **reran the frozen LLM each time** (O(N²), no KV cache) — the dominant slowness.
- **Backbone.** GPT-2 / Llama-2 only; hand-rolled positional encoding and special tokens; broke for
  chat models.
- **Result.** `n ≈ 10–15` concept vectors saturate accuracy; even `n = 1` beats text-RAG at ~130×
  fewer tokens. **This is the bar v2 must clear.**

v1 proved the *existence* of a good concept-token code. v2 is a principled rebuild aimed at a
stronger objective, a cleaner encoder, chat-model support, speed, and — most importantly — at the
question *"is it learning the graph, or just compressing the prompt?"* (§8).

---

## 4. v2 component 1 — Featurization (turning a subgraph into vectors)

**What.** Each edge `(r_i, o_i)` becomes one feature vector by embedding the **label strings** with
the frozen LLM's *input* embeddings and mean-pooling over subword tokens, then **concatenating**
relation and object: `f_i = [ meanpool(emb(r_i)) ; meanpool(emb(o_i)) ] ∈ ℝ^{2d}`. The center
entity's label is embedded separately as conditioning.

**Research lineage.** Inductive KG representation learning: BLP (2010.03496) and NodePiece
(2106.12144) show that an entity representation built from *text/local structure* generalizes to
**unseen** entities, unlike transductive lookup-table embeddings (TransE/RotatE/ComplEx), which have
no vector for a new entity (cold start) and live in a space unaligned with the LLM.

**Why sound.**
- *Inductive by construction.* Any entity/relation with a label can be embedded → no per-entity
  parameters, no cold start. Wikidata has 100M+ entities; a lookup table is a non-starter.
- *Pre-aligned.* Features live in the LLM's own embedding space, so the encoder's job is a
  *transformation within one space*, not bridging a modality gap (which BLIP-2/GNP needed extra
  machinery for).
- *Concat, not v1's additive fusion.* Adding the relation vector into the neighbor (`K = N + E`)
  lets a high-norm relation wash out neighbor identity. Concatenation keeps both recoverable; the
  encoder's input projection learns the mixing. (FiLM/gating is a documented upgrade.)

**Caveat.** Mean-pooling subwords is a lossy "bag of word-pieces" — it discards word order inside a
multi-token name. The encoder partly compensates, and it is a *fixed* feature (the LLM is frozen),
so this is a known crudeness, not a correctness bug.

---

## 5. v2 component 2 — The encoder (subgraph → `k` concept tokens)

**What.** A **Perceiver / Flamingo-style latent-query resampler**. `k` learned latent vectors
(the "queries", which *are* the concept tokens) repeatedly **cross-attend** over the set of `N` edge
features and emit exactly `k` outputs, independent of `N`. Each of `L` blocks is
`cross-attn(latents ← edges) → self-attn(latents) → FFN` (pre-norm). Internal width `d_model` is
**decoupled** from `d`; a single output projection maps `d_model → d`. (`model/encoder.py`.)

**Research lineage.** Perceiver/Perceiver-IO (latent bottleneck over large inputs), Flamingo's
Perceiver Resampler (2204.14198, variable visual features → fixed token count), BLIP-2's Q-Former
(2301.12597, learned queries + cross-attention → soft prompts for a frozen LLM). For the
graph-specific question: GraphToken (Perozzi et al. 2024) and "Talk like a Graph" (2310.04560).

**Why a resampler and not the obvious alternatives — with reasoning.**

- *Not an MLP projector (LLaVA-style).* An MLP needs fixed-size input; our input is a
  **variable-size set** of `N` edges. You'd have to pool first, destroying per-fact structure.

- *Not a GNN / message passing.* The neighborhood is a **depth-1 star**: a center plus neighbors,
  with no neighbor–neighbor edges. One round of message passing from neighbors to center is exactly
  one attention/aggregation step — which the resampler already does. Additional GNN layers
  re-aggregate the same set (redundant) or oversmooth. GraphToken's empirical headline: *no GNN
  architecture dominated; featurization mattered more than message-passing depth.* So the GNN
  inductive bias buys nothing for a star graph; a **set encoder** is the structurally correct match.

- *Not the full Q-Former.* BLIP-2's contrastive ITC/ITM/ITG objectives and text tower exist to
  *align an image encoder to language*. Our features are already in the LLM's space — there is no
  modality gap — so that machinery is over-engineering.

**Permutation invariance — the mathematically important property.** Graph neighbors are an
**unordered set**: a function of them must satisfy `f(π·{f_i}) = f({f_i})` for any permutation `π`.
Cross-attention with **no positional encoding on the keys** computes, for latent `q`,
`Σ_i softmax_i(⟨q,k_i⟩) v_i` — a sum over keys, invariant to their order. Deep Sets (1703.06114) and
Set Transformer (1810.00825) prove this class (elementwise encode → symmetric pool, optionally with
inducing-point self-attention = our learned latents) is a **universal approximator of permutation-
invariant set functions**. So our encoder is exactly permutation-invariant *by construction* — we
verified it numerically (reordering neighbors leaves the output identical to 1e-5). Contrast: if you
fed raw per-neighbor tokens directly into the LLM with sequential positions, you would inject a
**spurious order** the data doesn't have. We keep invariance in the encoder; the `k` *outputs* have
a canonical (learned) slot order, which is where RoPE positions are later applied (§6).

**Why decoupling `d_model` matters.** v1 tied the encoder width to `d_llm`. Decoupling lets you tune
encoder capacity independently of the (small) 0.6B model — wider/narrower reasoning width without
touching the interface. GraphToken found most useful capacity lived in the *projection head*, not
the graph body; the single `d_model→d` output FC is where we spend it.

---

## 6. v2 component 3 — Injection (putting concept tokens into the frozen model)

Two sub-problems: *how much* to inject (the gate) and *where* in position space (RoPE).

### 6.1 The zero-init gate

**What.** A per-concept-token scalar gate, `inject = tanh(g) ⊙ C`, with `g` **initialized to 0**.
At step 0, `tanh(0)=0`, so the injected vectors are zero. (`model/injection.py::ConceptGate`.)

**Research lineage.** LLaMA-Adapter (2303.16199): zero-initialized, tanh-gated attention so a frozen
model's behavior is *provably unchanged at init*, then the adaptation signal is ramped in.

**Why sound.** Splicing **random** large vectors into `inputs_embeds` at init would shove the frozen
model's activations off-distribution and produce a huge, noisy initial loss — possibly damaging the
very capability-preservation we want. Starting at the identity (injection = 0) and letting the
optimizer *grow* the intervention keeps training in-distribution throughout. This is also the
strongest "no-harm" regularizer: at init the student is maximally close to the untouched model.

**The dead-zone subtlety we discovered empirically (and the fix).** The gradient to the encoder is
`∂L/∂C = tanh(g) · ∂L/∂inject`. At `g≈0`, `tanh(g)≈0`, so **the encoder receives ~no gradient** —
and `∂L/∂g = sech²(g)·⟨C, ∂L/∂inject⟩` is small if `C` is small. With a single shared learning rate
(1e-4) both stayed stuck: KL crawled `1.05→0.53` over 150 steps, raw gate frozen at `±0.001`. **Fix:
give the gate its own parameter group at ~100× LR (`gate_lr=1e-2`).** The gate then opens to ~0.06
and stabilizes; concept norm settles ~1.55 (on the embedding manifold, not exploding); KL falls
`1.05→0.04` in 200 steps. This is a real, generalizable lesson: *a single input-level zero-init gate
must be on a fast LR schedule or it self-throttles.* (LLaMA-Adapter has many gates across layers, so
its signal accumulates; we have one, so it must move.)

### 6.2 Positional embeddings (RoPE)

Qwen3 uses **rotary** position embeddings: each token's key/query is rotated by an angle
proportional to its absolute position, so attention depends on **relative** position.

**Decision.** Give the `k` concept tokens a **contiguous, unit-step block** in exactly the slot the
verbalized-facts text occupied for the teacher, and let the question continue monotonically:
`[system @ 0..p][concepts @ p+1..p+k][question @ p+k+1..]`. We realize this with the standard
attention-mask cumsum (`build_position_ids`).

**Research lineage & empirics.** Qwen2-VL M-RoPE (2409.12191) and a Qwen positional study
(2510.23095) test the alternatives directly and find three **empirically harmful** failure modes,
all of which our choice avoids:
- a **gap** before the injected block ("reserving" the facts span) → the model *ignores* the
  injected content;
- **position overlap** between concept tokens and later/generated text → "modality confusion",
  repetition loops;
- **resetting/zeroing the text positions** → drops *below* the vanilla baseline.
A fourth tempting option — **all `k` share one position** — makes the tokens positionally
indistinguishable to the LLM (identical rotation) and is off-distribution; permutation symmetry
belongs in the *encoder* (§5), not the LLM interface.

**Why sound.** Matching the teacher's position layout keeps the student's *question* tokens at the
same RoPE offsets the teacher saw, so the per-position KL targets are computed under matched
conditioning. Contiguous unit-step positions are exactly what the pretrained model expects.

**Caveat.** RoPE attention decays with relative distance, so for *long* generations the model may
under-attend to the front-loaded concept block (Vista-LLaMA 2312.08870). For short QA this is
unlikely to bite; the documented fallback is a V2PE-style (2412.09616) fractional-position
compression of the block. Not yet needed.

---

## 7. v2 component 4 — The distillation objective (the heart)

### 7.1 Self-distillation, and why the target is realizable

**Same backbone for teacher and student.** Both are the frozen `M`. The teacher conditions on facts
*text*; the student on concept *tokens*. Because the model and tokenizer are identical, the student
**can in principle match the teacher exactly**: we are searching, over the continuous set
`C ∈ ℝ^{k×d}`, for a code that makes the *same decoder* produce the *same distribution* it produces
when reading the facts. The KL has a **realizable zero** in the limit of enough capacity.

This is why self-distillation is the correct framing and why **Gemma (the question generator) is not
the teacher**: a different model has a different vocabulary and representation, so its distribution
is not a reachable target for Qwen3's concept tokens. (Gemma only writes the questions.)

**Is `k` tokens enough? An information-capacity argument.** A typical neighborhood is a dozen
triples — a few hundred bits of information. The channel is `k·d` continuous reals (e.g.
`8·1024 = 8192` dimensions), and the decoder is a competent LLM. Capacity ≫ information, so a
near-lossless code plausibly exists; the empirical evidence (v1: `n=1` already beats text-RAG; ours:
`k=8 → KL 0.04` on the overfit set) supports this. The binding constraint is not capacity but
whether the *encoder can compute* the code and whether it *generalizes* (§8, §9).

### 7.2 Soft-target KL over the full distribution ("dark knowledge")

**What.** At each path position we minimize `KL(P_T ‖ P_S)` over the **entire** vocabulary, not just
the answer token. With temperature `τ`: soften both by `/τ` and scale the loss by `τ²` (Hinton KD).
The teacher is **detached** (a fixed target). (`train/losses.py::sequence_kl`.)

**Research lineage.** Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network"
(1503.02531): the teacher's *full softmax* encodes **dark knowledge** — the relative probabilities of
the non-argmax tokens — which carries the teacher's similarity structure. Matching it transfers far
more per position than matching a one-hot answer.

**Why sound / why it beats v1's hard CE.**
- *Denser signal.* Each position constrains the student across all `V` logits, not one — more
  gradient information per token, more sample-efficient.
- *Belief-state cloning, not answer-cloning.* The concept tokens must reproduce *how the
  facts-conditioned model distributes probability*, i.e. its reasoning state, not merely its top
  token. This is the difference between "knows the answer here" and "behaves like it has the facts."
- *Capability preservation as a structural property.* The target is a **benign** distribution: the
  frozen model reading facts under a neutral system prompt. So the concept tokens are pulled toward
  *normal in-context-facts behavior*, never toward hijacking the model. v1's hard label on the
  answer could, in principle, push the tokens to force an answer regardless of context. KL-to-a-
  benign-teacher is a safeguard by design.

**Forward vs reverse KL — a known limitation.** `KL(P_T‖P_S)` is **mass-covering**: the student is
penalized for putting low probability where the teacher has mass, so it must cover all of the
teacher's modes. With small `k` the student may be unable to cover everything and will **smear**
probability. Mode-seeking reverse KL or on-policy GKD (2306.13649) is the documented lever if this
bites.

### 7.3 Teacher-forcing along the stored greedy path

**What.** Offline (CF-Train Stage 5) we recorded the teacher's **greedy continuation** token-ids for
each example. At train time we teacher-force *both* teacher and student along that path and recompute
the teacher's per-position distribution live (cheap — same frozen model). The path is the behavior we
want to clone; teacher-forcing means both models see the same gold prefix at each step, so the KL is
computed on matched conditioning along the trajectory we care about.

**The alignment subtlety (and why it's a separate, tested function).** Teacher and student have
**different-length prefixes** (facts text vs `k` tokens), so the path sits at different absolute
indices in each sequence. In a causal LM the logits at position `t` predict token `t+1`, so path
token `j` (path starting after a length-`c` prefix) is predicted by the logits at index `c-1+j`. We
gather those positions into a compact `(B, m, V)` tensor for *each* of teacher and student — now
aligned by path index `j` and directly comparable. (`train/forcing.py::gather_path_logits`, unit-
tested with hand-computed indices.)

### 7.4 Capability preservation (no-harm), end to end

Three independent safeguards stack:
1. **Frozen weights** → no catastrophic forgetting is even possible.
2. **KL toward a benign facts-teacher** → the trained intervention is "behave as if you had these
   facts in context," not "hijack."
3. **Zero-init gate** → the student starts as (nearly) the untouched model and ramps in.
Plus the CF-Train **control task family** (subject mentioned, task *not* about its facts → target is
the model's *normal* behavior), which trains the concept tokens not to force subject-talk. The
**no-harm eval** (general/control tasks with vs without injection) is built but not yet run.

---

## 8. The central theoretical problem: compression vs. knowledge

This is the question the user pushed on, and it is the deepest part of the design.

**Two hypotheses for what the encoder learns.**
- **(a) Mere compression.** `C_e` is a lossless squeeze of the *verbalized facts it was shown* — a
  per-instance autoencoder. Generalizes nothing; it's a (good) prompt compressor.
- **(b) Genuine entity/graph representation.** `C_e` is a reusable entity embedding that (i)
  composes for multi-relation/multi-hop queries, (ii) is robust to *which* neighbors were shown, and
  (iii) places entities in a consistent relational space the frozen LLM can reason over.

**The uncomfortable theorem.** Consider the objective literally: distill, for each entity, the
teacher reading *one fixed verbalization*, on queries whose answers are present in that verbalization.
Then a **lossless code of the verbalization is a global optimum** — it makes the decoder reproduce
every needed answer. Nothing in the objective rewards cross-entity geometric consistency or
generalization to unshown facts. **So (a) is Bayes-optimal for the naive objective; (b) does not come
for free.** This is exactly what the prompt-compression literature optimizes: Gist tokens
(2304.08467), ICAE (2307.06945), 500xCompressor (2408.03094, explicit *reconstruction* loss), xRAG
(2405.13792, one-token document compression, frozen LLM, precomputed — and *no* compositional claim).

**Why local-only training can nonetheless yield (b) — the escape, and its single condition.**
Inductive KG embeddings (NodePiece 2106.12144, GraIL 1911.06962, ULTRA 2310.04562) prove that a
representation computed *purely from an entity's local neighborhood* can be reusable and globally
consistent on **unseen** entities — **iff** the model encodes entities as *functions of shared,
entity-agnostic relational structure* rather than memorized per-instance content. The lever that
makes this realistic here is the **frozen LLM**: it already holds a globally-consistent semantic
space. If the encoder places `C_e` *onto that manifold* in a way consistent with how the LLM
represents `e`'s type and relations, then "local training" is really **projection onto a pre-existing
global space**, and global consistency is **inherited from the frozen LLM**, not learned from a
global graph the encoder never sees. (Our overfit already shows concept norms settling on the
manifold (~1.55), which is encouraging but not yet proof of consistency.)

**The hard information-theoretic limit.** `C_e` is a function of `G_e` (1-hop). By the data-
processing inequality it contains **no information about 2-hop facts** not in `G_e`. So single-entity
tokens **cannot** answer a multi-hop query whose answer is not in the 1-hop neighborhood — full stop.
Multi-hop must be **inference-time composition**: splice `C_A` and `C_B` and let the frozen LLM chain
them (Guu path queries 1506.01094; Query2Box 2002.05969), with expected cascading error. The (b)
claim is therefore about *composability and consistent placement*, **not** "recovering unshown edges
from one entity."

**How we engineer toward (b) — design levers, ranked.**
1. **Distill a *distribution* of diverse queries per entity; never a reconstruction loss.** This is
   the single decision separating us from ICAE/500xCompressor. (CF-Train already mixes Gemma-diverse
   QA + compositional + descriptive + control tasks.)
2. **Neighbor-subsampling distillation.** Build the *teacher* from a **varying random subset** of the
   neighbors each step. The concept tokens must then be a *stable* entity representation invariant to
   which facts were verbalized — directly attacking the per-instance squeeze. (Cheap; high impact;
   *not yet implemented*.)
3. **Manifold regularization.** Pull `C_e` toward the statistics of the frozen `inputs_embeds`
   distribution → the inheritance mechanism above. (Empirically may be mild; concepts already landed
   near the manifold.)
4. **Cross-entity relational contrastive** (`C_Paris − C_France ≈ C_Berlin − C_Germany`) — impose
   relational geometry (RotatE/Query2Box-style; relational KD 1904.05068).
5. **Compositional / two-entity training** — the only way to *train* the inference-time composition
   we want to evaluate.

**How we will *prove* (b), not assume it — the discriminating evals.** The governing principle:
*find a query whose answer is not a substring of the shown facts, that an autoencoder cannot serve,
and show ConceptFormer answers it.*
- **Neighbor-invariance probe (decisive, cheap).** Build `C` from disjoint neighbor subsets `S1`,
  `S2` of the *same* entity. (a) predicts two different codecs; (b) predicts near-identical tokens
  that **answer each other's held-out questions**. High cross-subset accuracy is impossible for a
  pure autoencoder, and it sidesteps the "1-hop can't hold out edges" objection (the edge is held out
  from the *verbalization*, not the graph). This is the matched partner of lever 2.
- **Token-swap counterfactual** (splice `B`'s tokens into a prompt about `A` → does it answer about
  `B`?), **linear probe** to held-out attributes, **nearest-neighbor geometry** of the concept-token
  lookup table, **inductive unseen-entity probe**, **multi-hop composition** held-out.

---

## 9. Limitations, shortcomings, and risks (consolidated)

**Methodological / theoretical**
- **Compression-collapse (the big one).** Without levers 1–2 the optimum is an autoencoder (§8). We
  have the architecture but have *not yet* added neighbor-subsampling or run the discriminating
  probes. Until then, "it learns the graph" is a hypothesis, not a result.
- **1-hop information ceiling.** Single-entity tokens cannot encode 2-hop facts; multi-hop is only
  achievable by inference-time composition, with cascading error. Any multi-hop headline must be
  framed that way.
- **Self-distillation ceiling.** The student can at best match the teacher (frozen model + facts),
  which itself is imperfect (~94% PopQA; residual gold/metric noise, granularity mismatches).
  ConceptFormer inherits the teacher's errors — it cannot exceed RAG-with-facts on this objective.
- **Forward-KL smearing** with small `k` (§7.2); may require reverse-KL/GKD.
- **Teacher-hacking / fixed-dataset distillation** (2502.02671, 2106.05945): matching the teacher
  only on its training support; degenerate if one fixed verbalization per entity. Mitigated by
  diverse queries + subsampling, not yet by on-policy distillation.
- **High in-distribution accuracy ≠ knowledge.** Distillation can transfer task accuracy without the
  teacher's representation (2505.15442). Only the §8 held-out/swap/probe tests discriminate; raw
  PopQA accuracy does not.

**Architectural / mechanistic**
- **Graph-token "sink" collapse** (2606.03712): injected tokens can become high-attention outliers
  the LLM attends to but does not *use*. The KL-to-text-teacher objective guards against this (it
  forces functional equivalence), but we must verify with token-knockout / causal tracing.
- **Question-agnostic compression is hard.** The encoder sees the *whole* neighborhood without the
  question and must pack all `N` edges into `k` slots — strictly harder than KAPING/GNP, which filter
  triples by the question. **High-degree entities** (hundreds–thousands of edges) stress a fixed `k`
  far more than the popular-entity PopQA average suggests.
- **Lossy mean-pool featurization** (§4): multi-token names lose word order.
- **Gate fragility** (§6.1): a single input-level zero-init gate self-throttles without a fast LR; it
  is coarser than LLaMA-Adapter's per-layer gates and may need that upgrade at scale.
- **RoPE long-range decay** for long generations (§6.2); mitigations on the shelf, not applied.
- **Capability preservation unproven.** Three safeguards are in place; the no-harm eval is built but
  not yet run.

**Engineering / empirical (the caveats from the build)**
- **Overfit ≠ generalization.** Our `KL 1.05→0.04` is on **16 training examples**. It proves the
  function class has the *capacity* to carry a neighborhood and that the whole pipeline (featurize →
  encode → gate → splice → forward → KL) is wired correctly and trains. It says **nothing** about
  generalization to unseen entities. The immediate next test is held-out KL + PopQA accuracy via a
  `ConceptFormerPredictor` (which already slots into the eval harness + token accounting).
- **Per-example training loop.** The trainer currently processes one example at a time (correct, but
  slow). Batched/padded forward is the scaling step before any large run.
- **Data at smoke scale.** The full CF-Train corpus (~20k entities) is built and validated but not
  yet generated at scale; we are deliberately developing the trainer on the cheap smoke set first.

---

## 10. v1 → v2 at a glance

| Aspect | v1 (arXiv:2504.07624) | v2 | Why |
|---|---|---|---|
| Encoder | flat attn, `n` independent Q/K/V heads | `L`-layer latent-query resampler, shared `k` latents | cleaner, multi-layer refinement, less param waste |
| Internal dim | coupled to `d_llm` | **decoupled** `d_model` + output FC | tune capacity independent of the 0.6B model |
| Edge fusion | additive `N + E` | concat / FiLM | relation can't wash out neighbor |
| Permutation | implicit | explicit set-encoder invariance (verified) | neighbors are an unordered set |
| Injection | plain splice | **zero-init tanh gate** (fast LR) | no-harm at init + stable training |
| Positions | GPT-2 hand-rolled | contiguous unit-step RoPE block in the facts slot | chat-model native; avoids known RoPE failure modes |
| Objective | **hard CE** on the answer | **full-distribution KL** self-distillation | dark knowledge; belief-state cloning; capability-preserving |
| Speed | O(N²) rerun per target token, no KV cache | teacher-forced single forward; KV cache at inference | the v1 bottleneck removed |
| Backbone | GPT-2 / Llama-2 | Qwen3-0.6B (chat) | modern, validated 94% RAG ceiling |
| No-harm | none explicit (hard label could hijack) | KL-to-benign-teacher + control tasks + zero-init gate + planned no-harm eval | capability preservation as a first-class goal |

**The one-sentence difference.** v1 taught a graph encoder to make a frozen LM *emit the right answer
token* (hard CE); v2 teaches it to make the frozen LM *hold the same belief state it would if it had
read the facts* (full-distribution KL self-distillation), with a cleaner permutation-invariant
resampler, a no-harm zero-init injection, principled RoPE placement, and an explicit research program
to show it learns reusable entity representations rather than compressing prompts.

---

## 11. Empirical status (2026-06-14)

- **Proven:** the pipeline trains end to end; `k=8` concept tokens overfit 16 examples to `KL≈0.04`;
  the gate opens to a healthy operating point with concept norms on the manifold; the zero-init-gate
  dead-zone is understood and fixed (separate fast LR).
- **Not yet shown:** generalization (held-out KL + PopQA accuracy), the compression-vs-knowledge
  discrimination (neighbor-invariance probe etc.), no-harm preservation, behavior at data scale,
  batched-forward throughput.

---

### Verified arXiv ids
2504.07624 ConceptFormer v1 (Barmettler) · 2301.12597 BLIP-2 · 2303.16199 LLaMA-Adapter ·
2304.08467 Gist tokens · 2309.15427 Graph Neural Prompting · 2510.23095 multimodal positional (Qwen)
· 2106.12144 NodePiece · 1810.00825 Set Transformer · 2310.04560 Talk-like-a-Graph.
*Real but not on the HF Papers index:* 2204.14198 Flamingo · 2405.13792 xRAG · 2307.06945 ICAE ·
2408.03094 500xCompressor · 2310.04562 ULTRA · 1911.06962 GraIL · 2002.05969 Query2Box ·
1506.01094 Guu path queries · 1503.02531 Hinton KD · 2306.13649 GKD · 2502.02671 teacher-hacking ·
2010.03496 BLP · 2306.04136 KAPING · 2412.09616 V2PE · 2312.08870 Vista-LLaMA · 1703.06114 Deep Sets
· 2606.03712 graph-token sinks · 1904.05068 relational KD · 2505.15442 KD-not-representation ·
2409.12191 Qwen2-VL.
