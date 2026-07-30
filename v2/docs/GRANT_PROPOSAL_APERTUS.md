# Swiss AI Initiative compute grant -- DRAFT (Apertus-anchored, 2026-07-29)

> Working draft. Re-spined around 4 work packages with Apertus as the anchor (supersedes the
> scaling-law-framed GRANT_PROPOSAL_DRAFT.md). Format target: research plan <= 2 A4 pages
> (11pt, 1.5 spacing), excluding cover, references, appendix. Compute ask sized bottom-up in
> Appendix A. Authoritative version is grant/proposal.tex. Open: [page/format limit],
> [exact grant program + deadline], [Apertus arXiv citation].

---

## COVER PAGE (placeholders)

- **Title:** An Open, Multilingual, Multi-hop Knowledge Layer for Apertus
- **Scientific lead:** Joel Barmettler (researcher, University of Zurich)
- **Swiss PI (signs):** Prof. Dr. Abraham Bernstein (Department of Informatics, UZH)
- **Requested allocation:** ~6,000 GH200 GPU-hours over 6 months (bottom-up, Appendix A)
- **Backbones:** Apertus mini family v1.1 -- 0.5B/1.5B/4B-Instruct (primary) + Qwen3 / Gemma-3 (validated controls)
- **Open artifacts:** corpora (10k-1M entities, multilingual, multi-hop; CC0 source), all
  checkpoints, evaluation harness, per-item results dataset -- open weights, open data, open code.

---

## RESEARCH PLAN (2 pages)

### 1. Context and track record

Frozen language models reach factual knowledge two ways: from weights (uneditable, weak on the
long tail) or from retrieved text (hundreds of context tokens per query). ConceptFormer adds a
third channel: compress an entity's knowledge-graph neighborhood into a few continuous "concept
tokens" that a frozen model consumes in place of retrieved text. The predecessor system
(Barmettler, Bernstein & Rossetto) was published at **The Web Conference 2026** and received the
**best paper award** of its hosting workshop, showing on GPT-2 that latent concept injection
beats graph textification by up to +272% Hit@10 at ~130x fewer tokens.

Over the past months we rebuilt the method on modern instruction-tuned backbones and hardened
the evaluation (full 14,266-question PopQA on unseen entities, leakage-free held-out, Wilson
CIs, paired McNemar, 3 seeds). Established results, all reproducible from the public repo and
W&B:

- **Token efficiency.** Eight concept tokens lift a frozen Qwen3-0.6B from 0.10 to 0.48
  exact-match on unseen entities; text baselines need about five to six times more tokens for
  the same accuracy. An untrained-injection control (0.13) confirms the trained encoder is the
  effect.
- **Data scaling, not yet saturated.** 10x more training entities (10k -> 100k) raises
  unseen-entity accuracy 2.3x on identical eval sets (single-placement re-anchor 2026-07-30), with the token-budget curve still
  climbing at k=32.
- **The model reads the graph (causal).** Counterfactual edge-swap probes: the model follows a
  rewired edge to the false answer at a rate rising with the token budget, evidence current
  graph-token systems are shown to lack.
- **Cross-family and cross-graph generalization.** The recipe transfers from Qwen3 to Gemma-3
  without retuning (both families replicate the token/data trends), and a Wikidata-trained
  encoder transfers zero-shot to two unseen graphs (MetaQA movies, WorldCup2014 sports).

**The limits we hit, which define this proposal.** On 24 GB consumer GPUs we cannot go past
these first points: (i) corpora beyond 100k entities exceed our throughput, yet the data axis is
still rising; (ii) the encoder trained on English labels does not carry across the *label
language* -- German-labeled concepts stay near the no-injection floor (about 9% gap closure vs
42-50% for English), a boundary we located precisely but cannot yet cross at scale; (iii) our
concept tokens compress a 1-hop neighborhood, and multi-hop composition is only validated as a
small proof of concept. Each limit is a work package below, and each already has a first result.

### 2. Work packages

**WP1 -- Scale to one million entities.** Complete the data-scaling law from 100k to 300k
(partially built) and 1M Wikidata entities, on the primary Apertus-1.5-mini backbone. This is
the single best-evidenced extension: the ~2.3x gain per decade has not saturated, so the open
question is where the returns to training data level off, and whether 1M entities closes the
remaining gap to text-RAG. Deliverable: the first data-scaling law for an inductive knowledge
encoder, plus 300k/1M open corpora.

**WP2 -- Multilingual ConceptFormer.** Our cross-lingual study shows accuracy is set by the
concept *label* language, not the question or system language: an English-trained encoder leaves
non-English-labeled concepts near the floor. The fix is direct -- train the encoder on
target-language Wikidata labels -- but needs per-language corpora and teacher paths we cannot
generate at scale locally. We will train a multilingual encoder across a representative set of
languages (aligned with Apertus's multilingual coverage) and measure per-language unseen-entity
accuracy and answer-language fidelity. Deliverable: a multilingual knowledge layer, the natural
match for a multilingual open model.

**WP3 -- Multi-hop concept composition.** To answer the "this is just text compression"
critique, we represent each neighbor by its *own* concept vector rather than its label
embedding, so a single shared-weight encoder composes information across hops (1-hop concepts ->
2-hop -> 3-hop). Our proof of concept already learns 2- and 3-hop MetaQA answers that single-hop
concepts cannot, with warm-starting and in-domain label-free adaptation both helping. WP3 builds
the Wikidata multi-hop QA corpus and trains the recursive composition properly (label-free KL,
shared weights), turning ConceptFormer from a neighborhood compressor into a learned graph
embedder. Deliverable: multi-hop concept injection and the recursive-encoder recipe.

**WP4 -- Apertus-1.5-mini backbones.** Our recipe already transfers across model families
without retuning, so applying it to the Apertus mini family (v1.1 Instruct: 0.5B/1.5B/4B) is low-risk and
high-value: it produces the first knowledge-injection results on the Swiss open model, and lets
us run WP1-WP3 with Apertus as the backbone throughout. Deliverable: concept encoders and
open checkpoints for each Apertus-1.5-mini size, including the marquee combination -- a
multilingual, multi-hop concept layer on Apertus.

### 3. Fit with Apertus and open science

ConceptFormer's only knowledge source is **Wikidata (CC0, public domain)**; training is
**label-free** (self-distillation against the frozen model reading verbalized facts, no
proprietary QA); code, weights, corpora, and per-item eval outputs are released. This matches
Apertus's founding stance -- open data, open weights, open recipe -- exactly, and WP2 directly
serves Apertus's multilingual mission. A provenance-carrying, editable knowledge channel (every
concept token traces to KG edges, and counterfactual tests prove the model uses them) is a
hallucination-control primitive for open European models.

### 4. Timeline (6 months)

M1: 1M corpus generation (300k built) + Apertus-1.5-mini re-anchor on our validated recipe.
M2-M3: WP1 data transect (300k, 1M) + WP4 size sweep. M3-M4: WP2 per-language corpora, teacher
extraction, multilingual training. M4-M5: WP3 multi-hop corpus + recursive training. M5-M6:
cross-cutting evaluation (faithfulness, capability, per-language, multi-hop), law fitting, paper
+ artifact release. Workloads are independent single-node runs (embarrassingly parallel).

### 5. Deliverables

(1) A scaling + capability paper (target: top ML venue). (2) Open corpora: 300k/1M-entity,
multilingual, and multi-hop Wikidata QA-distillation sets (CC0 source, sha-manifested).
(3) All checkpoints on HuggingFace, including Apertus-1.5-mini encoders. (4) The open evaluation
harness. (5) The per-item results dataset for GPU-free re-analysis.

### 6. Data and ethics

Wikidata (CC0); PopQA and standard public benchmarks; questions generated by open-weight models
with released prompts. No personal data beyond public encyclopedic facts; no human subjects.

---

## APPENDIX A -- Compute, sized bottom-up

### A.1 Measured anchors (public W&B runs, RTX 4090 wall-clock)

- **Training.** One converged run at Qwen3-0.6B on the 100k-entity corpus (925k distill rows,
  100k steps, ~4 epochs, all trajectory evals) = **6.0 wall-hours on one RTX 4090** (W&B group
  `phaseC-kfamily-100k`). 4090->GH200 taken as **2x** (conservative: bf16 peak is ~6x, but the
  eval and small-batch phases do not saturate H100-class cores). Convergence margin **h = 1.4**
  (larger k/backbones still improving at the fixed horizon). Unit **U0 = 6.0/2 x 1.4 = 4.2
  GH200-h** at (0.8B, 100k entities); scales linearly in parameter ratio P and corpus ratio D.
- **Corpus prep** (tiering + teacher-path extraction): **6.8 GH200-h** per (0.8B, 100k).
- **QA generation** (open-weight model via vLLM): **~12 GH200-h** per 100k entities.
- **Definitive eval** per checkpoint (full PopQA + strict held-out + probes): **0.7 x P**.

### A.2 Work-package budget (GH200-hours; primary backbone P=1 unless noted)

| WP | content | arithmetic | hours |
|---|---|---|--:|
| 1 | 1M data law: gen 1M (12x10) + prep 1M (6.8x10) + prep 300k (6.8x3) + k{4,8,16,32}x3seeds at 300k (12x12.6) and 1M (12x42) + evals (24x0.7) | 120+68+20+151+504+17 | 880 |
| 2 | Multilingual: 12-lang QA gen (12x12) + per-lang teacher prep (12x6.8) + 1 multilingual encoder x3 + 6 monolingual x2 seeds (18x4.2) + evals (~60x0.7) | 144+82+76+42 | 344 |
| 3 | Multi-hop: 2-hop corpus gen (~240) + 2-hop prep (~100) + recursive hop2/hop3 k{8,16,32}x3 (18x4.2x1.5) + recursive-KL dev (50) + evals (18x0.7) | 240+100+113+50+13 | 516 |
| 4 | Apertus mini sweep (0.5/1.5/4B): per size 65.6xP; sum P = 0.63+1.88+5.0 = 7.5 + marquee corners (multilingual + 1M on 4B) | 65.6 x 7.5 + ~250 | 742 |
| 5 | Cross-cutting eval (faithfulness/capability/transfer/per-language across ~120 checkpoints) | ~120 x 0.7 x avg-P | 150 |
| | **Core subtotal** | | **~2,630** |
| | Contingency 25% (restarts, new-cluster debugging, our run history motivates this) | | 660 |
| | **Base plan** | | **~3,300** |
| | Stretch (named): 3M-entity point on primary (~940); full multilingual x multi-hop x Apertus corner (~400); +8 languages in WP2 (~300); double-horizon convergence check (~200) | | ~1,840 |
| | **Requested** | | **~5,000-6,000** |

The margin over the itemized total absorbs the one real uncertainty, the 4090->GH200 conversion
(2x vs 3x moves the whole budget +/-30%). **Fallback at ~2,500 h:** drop the stretch tier and
the 1M x Apertus corner; the four WPs' core results survive.

### A.3 Feasibility

Predecessor peer-reviewed and best-paper-awarded at WWW Companion '26. Public repo
(github.com/joelbarmettlerUZH/ConceptFormer): resumable, cached, CI-gated pipeline (corpus
generation via vLLM, teacher extraction, training, evaluation); every cited number traces to a
W&B run-id and per-item eval dumps. The recipe already runs across two backbone families without
retuning, so the Apertus migration is a re-anchor, not a redesign. Workloads decompose into
independent single-node jobs (high cluster utilization).

## APPENDIX B -- References

ConceptFormer v1: Barmettler, Bernstein & Rossetto, "ConceptFormer: Towards Graph-Native
Grounding of Large Language Models via Latent Concept Injection", WWW Companion '26, pp.
587-596, DOI 10.1145/3774905.3794653 (best paper award, hosting workshop; arXiv 2504.07624).
Apertus: Swiss AI Initiative, Apertus mini v1.1 (0.5B/1.5B/4B), huggingface.co/collections/swiss-ai/apertus-mini [fill technical-report/arXiv citation]. Related methods and
full survey: repo docs/RELATED_WORK.md.
