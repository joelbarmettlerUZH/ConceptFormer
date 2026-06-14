# ConceptFormer v2

Next iteration of ConceptFormer: encode an entity's knowledge-graph neighborhood into a
few **soft concept tokens** injected into a **frozen** LLM's input-embedding space, trained
by **distilling from a graph-in-context teacher**. Eval on **PopQA** + **EntityQuestions**.

Design docs: `../V2_DESIGN.md` (architecture/objective), `../V2_DATASETS.md` (data strategy).

## Setup

```bash
cd v2
uv sync                      # creates .venv, installs the data-layer deps
```

Point large artifacts at the big disk (the dev box's `/home` is nearly full):

```bash
export CF_DATA_ROOT=/data/conceptformer   # or any path on the 3.2 TB / partition
```

## Phase-1 data pipeline (current)

```bash
uv run conceptformer entity Q42                 # sanity: print a 1-hop neighborhood
uv run conceptformer popqa-info                 # load PopQA, show normalized rows
uv run conceptformer build-popqa-snapshot --limit 100   # snapshot PopQA subjects
```

Snapshots are versioned, content-hashed JSONL (`<CF_DATA_ROOT>/snapshots/<name>/`) — the
immutable graph artifact the rest of the pipeline consumes. The Wikidata source is the
public `wbgetentities` API for the pilot; swapping to a local qEndpoint later changes only
`WikidataClient._get_entities_raw`.

## Tests

```bash
uv run pytest -m "not integration"   # unit tests (offline)
uv run pytest -m integration         # hits the live Wikidata API
```

## Layout

```
src/conceptformer/
  config.py            # pydantic-settings (CF_* env vars)
  schemas.py           # Entity / Edge / Subgraph / QAExample
  data/
    wikidata.py        # endpoint-agnostic 1-hop truthy neighborhood extractor (+ sqlite cache)
    benchmarks.py      # PopQA (done) / EntityQuestions (todo) loaders → QAExample
    snapshot.py        # versioned, hashed snapshot builder
  cli.py               # typer CLI
```

Training/model code (Q-Former encoder, Qwen3 adapter, distillation trainer) lands in later
phases; data-layer deps are kept separate so this installs light.
