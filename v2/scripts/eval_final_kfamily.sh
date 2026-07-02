#!/usr/bin/env bash
# Post-fix definitive re-eval of the Phase-C k-family: FULL PopQA (~14k) + strict
# (fact-leakage-free) held-out, per-item dumps for paired stats — replaces the seed-coupled
# n=200 numbers behind F12 / GRANT_ANALYSIS. Plus the new baselines. One chain per GPU:
#   scripts/eval_final_kfamily.sh 0   # seed-0 + s1(k1/k2/k4) checkpoints, then untrained baselines
#   scripts/eval_final_kfamily.sh 1   # remaining seeds, then RAG retrieval curves (3 modes)
# Base/RAG text brackets are cached in generations.sqlite (WAL), so the first checkpoint pays
# them once and every later one only runs the concept pass.
set -u
cd "$(dirname "$0")/.."
GPU="$1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
CLI="uv run --group infer python -m conceptformer.cli"

run() { echo "=== $(date +%H:%M:%S) $*"; $CLI "$@" --device cuda:0 || echo "!!! FAILED: $*"; }

if [ "$GPU" = "0" ]; then
  for c in pc_k1_best pc_k2_best pc_k4_best pc_k8_best pc_k16_best pc_k32_best \
           pc_k1_s1_best pc_k2_s1_best pc_k4_s1_best; do
    run eval-final --checkpoint "$c"
  done
  # The no-encoder control at the two candidate recipe k's (same frozen sets -> paired).
  run eval-untrained-injection --k 8
  run eval-untrained-injection --k 16
else
  for c in pc_k8_s1_best pc_k16_s1_best pc_k32_s1_best \
           pc_k1_s2_best pc_k2_s2_best pc_k4_s2_best \
           pc_k8_s2_best pc_k16_s2_best pc_k32_s2_best; do
    run eval-final --checkpoint "$c"
  done
  # RAG-vs-budget baseline under all three retrieval modes, on the SAME strict/frozen sets.
  for mode in pagerank question summary; do
    run cf-rag-budget-curve --retrieval "$mode" \
      --out "data/analysis/rag_budget_curve_${mode}.json"
  done
fi
echo "=== chain done $(date)"
