#!/usr/bin/env bash
# 300k stepping-stone STEP 2 — Gemma generate + tier + extract on the cftrain_300k snapshot
# (259,600 usable entities; 28.8% pass on the deeper PageRank tail). ALL on GPU1 (GPU0 is seeding
# for ~3 days). Lifecycle: start vLLM/Gemma on GPU1 → generate (resumable per-entity cache) → kill
# vLLM to free GPU1 → tier + extract with expandable_segments. Mirrors the validated 100k pipeline.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2
NAME=cftrain_qa_300k
SNAP=cftrain_300k

echo "=== start vLLM/Gemma on GPU1 $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 uv run --with vllm vllm serve cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit \
  --port 8000 --gpu-memory-utilization 0.92 --max-model-len 8192 > /tmp/cf_vllm300.log 2>&1 &
VLLM_PID=$!
echo "vLLM pid=$VLLM_PID; waiting for ready…"
until curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null | grep -q gemma; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died — see /tmp/cf_vllm300.log"; exit 1; fi
  sleep 15
done
echo "=== [1/3] generate-cftrain $(date -u +%FT%TZ) ==="
uv run --group infer python -m conceptformer.cli generate-cftrain \
  --snapshot $SNAP --name $NAME \
  --n-questions 8 --descriptive 2 --control 2 --base-url http://localhost:8000/v1
GEN_RC=$?
echo "generate rc=$GEN_RC; stopping vLLM (free GPU1 for tier/extract)…"
kill "$VLLM_PID" 2>/dev/null; sleep 8
[ "$GEN_RC" -ne 0 ] && { echo "GEN FAILED rc=$GEN_RC"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== [2/3] tier-cftrain $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 uv run --group infer python -m conceptformer.cli tier-cftrain \
  --dataset $NAME --snapshot $SNAP --keep-easy 0.3 --device cuda:0 --batch-size 64
echo "=== [3/3] extract-teacher-paths $(date -u +%FT%TZ) ==="
CUDA_VISIBLE_DEVICES=1 uv run --group infer python -m conceptformer.cli extract-teacher-paths \
  --dataset $NAME --snapshot $SNAP --device cuda:0 --batch-size 32

echo "=== 300k CORPUS DONE $(date -u +%FT%TZ) — manifest ==="
cat data/cf_train/$NAME/manifest.json
echo "GEN300K_FULL_DONE"
