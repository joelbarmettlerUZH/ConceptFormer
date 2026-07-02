#!/usr/bin/env bash
# 300k scaling stepping-stone — STEP 1: build the snapshot (select entities + fetch 1-hop graphs).
# Network-bound (Wikidata), no GPU, resumable via wikidata_cache.sqlite — runs alongside GPU0 seeding.
# 100k needed 261,475 candidates @ 38% pass → for 300k usable select ~900k candidates (margin).
# Mirrors the 100k build (min_edges 6, PageRank neighbor ranking). PopQA eval entities are excluded by
# select-entities, so PopQA stays a clean cross-scale benchmark. (Gemma gen + tier/extract = step 2,
# after this completes, on GPU1.)
set -eu
cd /home/joelbarmettler/projects/ConceptFormer/v2

echo "=== select-entities (900k candidates) $(date -u +%FT%TZ) ==="
uv run --group infer python -m conceptformer.cli select-entities \
  --n 900000 --name cf_train_300k --seed 0

echo "=== build-cftrain-snapshot (target 300k usable) $(date -u +%FT%TZ) ==="
uv run --group infer python -m conceptformer.cli build-cftrain-snapshot \
  --candidates cf_train_300k --name cftrain_300k --target 300000 --min-edges 6

echo "=== SNAPSHOT DONE $(date -u +%FT%TZ) ==="
cat data/snapshots/cftrain_300k/manifest.json
echo "GEN300K_SNAPSHOT_DONE"
