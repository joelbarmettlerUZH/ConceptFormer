#!/usr/bin/env bash
# Phase F — graph-faithfulness across the whole k-family (best-seed best-checkpoint per k) on GPU1.
# Does the model READ THE GRAPH, and does faithfulness grow with k? Uses pc_k{k}_s1_best (s1 = best
# seed for every k). Captures each run's output to /tmp/cf_gf_k{k}.log for parsing.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2
for k in 1 2 4 8 16 32; do
  echo "=== GF k=$k ($(date -u +%FT%TZ)) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
    uv run --group infer python -m conceptformer.cli cf-graph-faithfulness \
      --checkpoint pc_k${k}_s1_best --dataset cftrain_qa_100k --snapshot cftrain_100k \
      --n 400 --seed 0 --device cuda:0 > /tmp/cf_gf_k${k}.log 2>&1
  echo "=== GF k=$k DONE exit $? ==="
done
echo "GF_SWEEP_DONE"
