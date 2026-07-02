#!/usr/bin/env bash
# 100k corpus FULL generation (committed). Same pipeline the smoke test validated, no --limit.
# Mirrors the 10k recipe (8 QA + 2 desc + 2 control/entity). Requires vLLM/Gemma up at :8000.
# generate (Gemma/vLLM on GPU0) is resumable via generations.sqlite; tier + extract run on GPU1.
# Projected ~930k distill rows. Long-running (~many hours) — launch in background.
set -eu
cd /home/joelbarmettler/projects/ConceptFormer/v2
NAME=cftrain_qa_100k
SNAP=cftrain_100k

echo "=== [1/3] generate-cftrain (100k entities) $(date -u +%FT%TZ) ==="
uv run --group infer python -m conceptformer.cli generate-cftrain \
  --snapshot $SNAP --name $NAME \
  --n-questions 8 --descriptive 2 --control 2 --base-url http://localhost:8000/v1

echo "=== [2/3] tier-cftrain $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 uv run --group infer python -m conceptformer.cli tier-cftrain \
  --dataset $NAME --snapshot $SNAP --keep-easy 0.3 --device cuda:0

echo "=== [3/3] extract-teacher-paths $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 uv run --group infer python -m conceptformer.cli extract-teacher-paths \
  --dataset $NAME --snapshot $SNAP --device cuda:0

echo "=== FULL GEN DONE $(date -u +%FT%TZ) — manifest ==="
cat data/cf_train/$NAME/manifest.json
echo "GEN100K_FULL_DONE"
