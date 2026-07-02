#!/usr/bin/env bash
# Phase F — capability preservation across the whole k-family (best-seed best-checkpoint per k) on
# GPU0. Do concept tokens DEGRADE normal generation on CONTROL tasks (entity named, but task not
# about its facts)? Reports greedy-agreement vs base + next-token KL(base||concept). Higher k = more
# concept tokens spliced in = the stress test for preservation. Logs to /tmp/cf_cap_k{k}.log.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2
for k in 1 2 4 8 16 32; do
  echo "=== CAP k=$k ($(date -u +%FT%TZ)) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    uv run --group infer python -m conceptformer.cli cf-capability-preservation \
      --checkpoint pc_k${k}_s1_best --dataset cftrain_qa_100k --snapshot cftrain_100k \
      --n 400 --seed 0 --device cuda:0 > /tmp/cf_cap_k${k}.log 2>&1
  echo "=== CAP k=$k DONE exit $? ==="
done
echo "CAP_SWEEP_DONE"
