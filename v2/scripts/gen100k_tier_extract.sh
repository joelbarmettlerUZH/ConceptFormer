#!/usr/bin/env bash
# 100k generation stages 2-3 ONLY (generation done: qa.jsonl = 1.1M rows from 91,200 entities).
# Stage-2 tier OOM'd in the original run because PYTORCH_CUDA_ALLOC_CONF was set only for the gen
# stage, not here — fragmentation killed it over the 1.1M-row pass. Fix: set expandable_segments AND
# run on a fully free GPU0 (vLLM/Gemma is shut down — not needed once generation is complete).
# tier reads qa.jsonl -> qa_tiered.jsonl; extract reads qa_tiered.jsonl -> qa_distill.jsonl.
set -eu
cd /home/joelbarmettler/projects/ConceptFormer/v2
NAME=cftrain_qa_100k
SNAP=cftrain_100k
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [2/3] tier-cftrain $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=0 uv run --group infer python -m conceptformer.cli tier-cftrain \
  --dataset $NAME --snapshot $SNAP --keep-easy 0.3 --device cuda:0 --batch-size 64

echo "=== [3/3] extract-teacher-paths $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=0 uv run --group infer python -m conceptformer.cli extract-teacher-paths \
  --dataset $NAME --snapshot $SNAP --device cuda:0 --batch-size 32

echo "=== TIER+EXTRACT DONE $(date -u +%FT%TZ) — manifest ==="
cat data/cf_train/$NAME/manifest.json
echo "TIER_EXTRACT_DONE"
