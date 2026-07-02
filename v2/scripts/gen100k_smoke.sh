#!/usr/bin/env bash
# 100k generation SMOKE TEST — validate the full pipeline end-to-end on a small slice of the staged
# cftrain_100k snapshot BEFORE committing the ~24h full run. Mirrors the 10k recipe (8 QA + 2 desc +
# 2 control per entity). Requires the vLLM/Gemma server up at :8000 (gen stage only).
# Stages: generate-cftrain (Gemma/vLLM) -> tier-cftrain (Qwen base/RAG signal) -> extract-teacher-paths.
set -eu
cd /home/joelbarmettler/projects/ConceptFormer/v2
N=${1:-150}
NAME=cftrain_qa_100k_smoke
SNAP=cftrain_100k

echo "=== [1/3] generate-cftrain ($N entities) ==="
uv run --group infer python -m conceptformer.cli generate-cftrain \
  --snapshot $SNAP --name $NAME --limit $N \
  --n-questions 8 --descriptive 2 --control 2 --base-url http://localhost:8000/v1

echo "=== [2/3] tier-cftrain ==="
CUDA_VISIBLE_DEVICES=1 uv run --group infer python -m conceptformer.cli tier-cftrain \
  --dataset $NAME --snapshot $SNAP --keep-easy 0.3 --device cuda:0

echo "=== [3/3] extract-teacher-paths ==="
CUDA_VISIBLE_DEVICES=1 uv run --group infer python -m conceptformer.cli extract-teacher-paths \
  --dataset $NAME --snapshot $SNAP --device cuda:0

echo "=== SMOKE DONE — manifest ==="
cat data/cf_train/$NAME/manifest.json
echo "SMOKE_PIPELINE_DONE"
