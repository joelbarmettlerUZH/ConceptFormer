# ConceptFormer-2

Graph-native grounding of frozen LLMs: encode an entity's Wikidata 1-hop neighborhood into
`k` continuous **concept tokens** that a frozen, instruction-tuned model consumes in place of
verbalized facts. The encoder is a small Perceiver-style resampler over the frozen model's own
label embeddings (inductive: an unseen entity needs one forward pass, no entity-specific
parameters), trained **label-free** by full-vocabulary KL self-distillation against the same
frozen model reading the facts as text.

Successor to ConceptFormer (WWW Companion '26, best paper award of its hosting workshop,
[DOI 10.1145/3774905.3794653](https://doi.org/10.1145/3774905.3794653)). The ConceptFormer-2
paper lives in `paper/` (in preparation for submission).

Headline (frozen Qwen3-0.6B, full 14,266-question PopQA, unseen entities, 3 seeds): eight
concept tokens lift exact-match accuracy from 0.10 to 0.48; text baselines need about five to
six times more tokens for the same accuracy. Counterfactual probes show the model reads the
injected graph, and the encoder transfers zero-shot to knowledge graphs it never saw.

## Setup

```bash
uv sync                    # lightweight data layer
uv sync --group infer      # + torch / transformers / wandb (training + eval)
```

Two dependency groups: the bare install runs the offline data tooling; `infer` pulls the GPU
stack. Figures need `--group viz`. All commands below run from this directory.

## Pipeline (offline data -> training -> evaluation)

```bash
# 1. Data: select entities, snapshot 1-hop neighborhoods, generate + tier QA, store teacher paths
uv run conceptformer select-entities --target 10000
uv run conceptformer build-cftrain-snapshot --name cftrain_10k
uv run conceptformer generate-cftrain --dataset cftrain_qa_10k
uv run conceptformer tier-cftrain --dataset cftrain_qa_10k
uv run conceptformer extract-teacher-paths --dataset cftrain_qa_10k

# 2. Train (one 24 GB GPU per run; 100k corpora need --no-cache-teacher)
uv run --group infer conceptformer cf-train \
  --dataset cftrain_qa_10k --snapshot cftrain_10k \
  --k 8 --batch 16 --grad-accum 2 --gate-mode none --placement before_entity \
  --steps 72000 --seed 0 --wandb --checkpoint my_k8_s0

# 3. Definitive evaluation (full PopQA + strict held-out, Wilson CIs, per-item dumps)
uv run --group infer conceptformer eval-final --checkpoint my_k8_s0
```

Reported configurations use 3 seeds (mean +/- std). Evaluation sets are frozen and shared
across checkpoints, so per-item dumps pair across runs for exact McNemar tests.

## Beyond the home graph

```bash
# Zero-shot cross-graph transfer (MetaQA movies / WorldCup2014 sports)
uv run --group infer conceptformer eval-transfer --checkpoint my_k8_s0 \
  --qa data/snapshots/metaqa/qa_test.txt --snapshot metaqa --source metaqa --n 0

# Cross-lingual (German PopQA; concept vectors from German vs English Wikidata labels)
uv run --group infer conceptformer eval-multilingual --checkpoint my_k8_s0 \
  --qa data/analysis/multilingual/popqa_de.jsonl --snapshot popqa_full_de \
  --mention localized --system-lang de

# Causal probes (edge ablation, counterfactual swap, capability preservation)
uv run --group infer conceptformer cf-graph-faithfulness --checkpoint my_k8_s0
uv run --group infer conceptformer cf-capability-preservation --checkpoint my_k8_s0

# In-domain adaptation / recursive multi-hop composition
uv run --group infer python scripts/recursive_hop.py --hop 1 --objective kl \
  --init-encoder my_k8_s0 --k 8
```

## Figures

```bash
uv run --group viz python scripts/cf2_figures.py                    # paper figures (scaling, probes, transfer, multilingual)
uv run --group viz python scripts/phaseF_token_efficiency_figure.py # token-efficiency figure
```

Figures read exclusively from `data/analysis/` (the released per-item eval dumps), so every
plotted number is reproducible without a GPU.

## Tests and code quality

```bash
uv run pytest -q -m "not integration"   # offline unit tests
uv run ruff check && uv run ty check    # lint + types (kept green)
```

## Repository layout

```
src/conceptformer/   library: data pipeline, model (featurizer/encoder/injection), train, eval
scripts/             experiment drivers and figure generation
paper/               the ConceptFormer-2 paper (LaTeX + figures)
grant/               Swiss AI Initiative compute-grant proposal
data/                machine-local artifacts (git-ignored): snapshots, corpora, checkpoints,
                     analysis outputs (eval_final/, transfer/, multilingual/, probes/, ...)
```

## Artifacts

- **W&B**: every training run logs to `university-of-zurich/conceptformer-v2`; checkpoints are
  downloadable `model:<name>` artifacts.
- **Hugging Face**: analysis outputs (per-item eval dumps backing every table and figure) are
  mirrored to [`joelbarmettler/conceptformer-data`](https://huggingface.co/datasets/joelbarmettler/conceptformer-data).
- **Data sources**: Wikidata (CC0), PopQA, MetaQA (CC BY), WorldCup2014.

## Citation

```bibtex
@inproceedings{barmettler2026conceptformer,
  title     = {ConceptFormer: Towards Graph-Native Grounding of Large Language Models
               via Latent Concept Injection},
  author    = {Barmettler, Joel and Bernstein, Abraham and Rossetto, Luca},
  booktitle = {Companion Proceedings of the ACM Web Conference (WWW Companion '26)},
  pages     = {587--596},
  year      = {2026},
  doi       = {10.1145/3774905.3794653}
}
```

MIT license.
