# ConceptFormer v2 — Dataset Design

Status: **draft for alignment**. Companion to `V2_DESIGN.md`. Locks the data strategy given the decisions: **eval on PopQA (primary) + EntityQuestions (twin)**, **Wikidata graph**, **Qwen3 backbone**, **distillation-from-graph-in-context objective**, phase-1 hardware **2×RTX 4090 (24 GB), context 2048–4096**.

---

## 1. The central constraint: eval sets are eval-only

PopQA (~14k) and EntityQuestions (~22k test) are **benchmarks, not training sets**. They are:
- small, and
- the thing we must show improvement on → training on them = leakage.

Therefore v2 has **two distinct data layers**:

| Layer | Purpose | Source | Touched at train time? |
|---|---|---|---|
| **CF-Train** (ours) | Teach the encoder: 1-hop neighborhood → soft tokens that answer s+r→o | Wikidata triples we sample + templated QA | **Yes** |
| **CF-Eval** | Measure transfer / headline numbers | PopQA + EntityQuestions (as released) | **No — frozen, never trained on** |

The scientific claim becomes: *a ConceptFormer trained on generic Wikidata QA transfers to held-out public benchmarks*, recovering most of the graph-in-context (RAG) gain at `k` tokens. This is exactly what reviewers asked for.

## 2. What PopQA / EntityQuestions actually give us (so we mirror them)

- **PopQA**: 16 relations (occupation, place of birth, genre, father, mother, capital, capital_of, country, producer, director, screenwriter, author, composer, color, religion, sport). Each row = gold `(subject QID, relation, object QID)` + templated question + subject Wikipedia popularity (pageviews). **Gold subject QID provided → no entity linking needed.** Long-tail subset (pageviews < ~100/mo) is the money figure.
- **EntityQuestions**: ~24 single-relation templates over Wikidata (born in, capital of, author of, …), entity-rich, designed so external knowledge is required. Gold entity provided.

Implication: CF-Train must **cover these relations** (plus more for generalization), over a **wide popularity range incl. long-tail**, in **QA/chat format**.

## 3. CF-Train construction

### 3.1 Triple sampling
- Sample `(subject, relation, object)` triples from Wikidata, **stratified by**: (a) relation — cover all PopQA+EQ relations + a broader set for generalization; (b) subject popularity — deliberately oversample long-tail entities (that's where CF beats parametric memory).
- **Leakage control (hard rule):** exclude every subject entity that appears in PopQA or EntityQuestions test from CF-Train. (Stronger than excluding just (subject,relation) pairs — avoids any subject-level memorization.)
- Target scale: order **100k–1M** training triples (tunable); start small (Lite) for the pilot.

### 3.2 Question format
- **Primary: relation-specific templates** (deterministic, cheap, no LLM) mirroring PopQA/EQ style, e.g. `"Q: In what city was {subject} born?\nA:"` → object. Chat-templated for Qwen3 (system + user turn), with the subject's **concept tokens spliced into a sentinel slot** in the user turn.
- **Optional augmentation: LLM paraphrases** of each template (a few natural variants per relation) so the encoder doesn't overfit one surface form. Decision pending (§7) — templated-only is a valid, simpler v1.
- This **replaces v1's Mistral-7B TriREx synthetic-sentence subsystem** with templated QA. Much simpler/cheaper/deterministic. (A declarative-sentence LM pretraining stage, TRExBite-style, remains an option but is not the default.)

### 3.3 The subgraph per example
For each subject we attach its **1-hop Wikidata neighborhood**: list of `(relation, neighbor)` edges, ranked by neighbor PageRank, capped at `N` (e.g. 100). The gold object is (by construction) one of these neighbors; the neighborhood also contains **distractors**, forcing the encoder to use the *relation in the question* to select — not just copy the only neighbor. Each entity/relation carries `label` (+ optional `description`) for the encoder's verbalized input (§5).

## 4. The Wikidata graph snapshot

- **Freeze once, version, hash.** Extract neighborhoods for the union of {CF-Train subjects, PopQA subjects, EntityQuestions subjects} and store as an immutable artifact (Parquet/safetensors). **No live SPARQL or AWS PageRank at train/eval time** (kills v1's biggest reproducibility wound).
- Fields per entity: QID, label, description, PageRank, list of edges `(relation QID, relation label, neighbor QID, neighbor label, neighbor PageRank)`.
- Pin the PageRank snapshot **as data**, not a hardcoded filename.
- Extraction mechanism is a §7 decision (live SPARQL one-off vs. Wikidata dump vs. local qEndpoint mirror) — the scale (tens of thousands of eval entities + up to ~1M train entities) makes live public SPARQL painful.

## 5. Pydantic schemas (typed, validated)

```
Entity   { qid, label, description?, pagerank }
Edge     { relation_qid, relation_label, neighbor: Entity }
Subgraph { center: Entity, edges: list[Edge] }          # capped + rank-sorted
QAExample{ subject_qid, relation, question, answer_qid,
           answer_aliases: list[str], subgraph_ref, popularity?, split }
```
Replaces v1's dict-juggling + CSV/TAR HF builder scripts. Snapshots are content-addressed.

## 6. Distillation-teacher data (ties to the decided objective)

For each CF-Train example we need a **teacher target**:
- **Teacher prompt** = same question with the neighborhood **verbalized as text in context** (or just the gold triple) — the "graph-in-context / RAG upper bound."
- Cache the teacher's **next-token distribution (and optionally hidden states) over the answer span** so KL-distillation doesn't recompute the teacher every epoch.
- **Context budget matters on 2×4090@2048:** the *student* is always `k` tokens regardless of neighborhood size, but the *teacher's* verbalized neighborhood grows with `N`. So cap the teacher's verbalized neighbors (e.g. top-M by PageRank, or gold triple + a few distractors) to fit 2048/4096. Record M so the teacher is reproducible. (This asymmetry — student is context-cheap, teacher is context-bound — is itself part of the efficiency story.)

## 7. Eval protocol (CF-Eval)
- Metric = the benchmark's standard: **accuracy / EM with alias-aware matching** (PopQA ships answer aliases; use them). Report overall **and** PopQA long-tail subset.
- Three bars, same backbone, same metric: **base LLM (no knowledge)** → **ConceptFormer (`k` tokens)** → **graph-in-context (teacher/RAG upper bound)**.
- Sweep `k` (concept tokens) → the accuracy-vs-tokens curve is the headline novelty.
- Backbone sweep: Qwen3-0.6B (efficiency story) and ≥1.7B (headline), all frozen — even 4B frozen (~8 GB bf16) fits a single 4090 for forward passes since only the small encoder trains; 0.6B/1.7B preferred for phase-1 throughput.

## 8. What v1 data machinery is dropped
TriREx Mistral-7B generation, fuzzy/`string.index` char-boundary matching (templated QA gives exact spans), live SPARQL + AWS PageRank at runtime, custom HF builder + TAR artifacts, the in-RAM 5M embedding cache. TREx→TRExStar **neighborhood-extraction idea** is kept (as a one-off snapshot builder); WebQSP path is parked unless we add it later.

## 9. Resolved decisions (2026-06-14)
1. **Training-question generation:** **templated + LLM paraphrases** — relation templates for exact spans + a few LLM paraphrase variants per relation for surface-form robustness and better transfer to PopQA phrasing.
2. **Wikidata snapshot mechanism:** **Wikidata `wbgetentities` REST API for the pilot** (validated: one batched GET per ≤50 ids → full truthy 1-hop claims + en labels; Q42 → 61 item-edges). The qEndpoint prebuilt-HDT mirror is permanently dead and no prebuilt HDT is hosted anywhere; the local-mirror option (build HDT from an NT dump via base `qacompany/qendpoint` + `/api/endpoint/load`) is **deferred** to the ~1M scale-up — see `conceptformer-v2-infra` memory. Extractor is endpoint-agnostic so API→local-qEndpoint is a one-line swap later.
3. **Pilot scale:** **Lite slice ~10–50k CF-Train triples**, PopQA/EQ relations only, to validate the full pipeline + distillation objective end-to-end on 2×4090 before scaling toward ~1M.

## 10. Hardware / storage constraints (this server)
- `/` (root): **3.2 TB free** → host the Wikidata mirror, teacher caches, model downloads, embedding snapshots here.
- `/home`: **only ~145 GB free (96% full)** — the repo lives here; keep large artifacts OFF it.
- **Redirect all big-data paths to `/` (e.g. `/data/conceptformer`)**: set `HF_HOME` / `HF_DATASETS_CACHE`, W&B dir, checkpoint dir, and the qEndpoint index dir, or `/home` fills mid-run.
- RAM 124 GB, 32 cores, 2× RTX 4090 24 GB. Backbone frozen → even Qwen3-4B fits forward passes; 0.6B/1.7B preferred for phase-1 throughput. Context 2048–4096 → cap teacher-verbalized neighbors accordingly (§6).
