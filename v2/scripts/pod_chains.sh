#!/usr/bin/env bash
# One sequential training chain per pod GPU for the v1.5 scaling grid. Usage:
#   CUDA_VISIBLE_DEVICES=N nohup bash scripts/pod_chains.sh <chain-id> > /workspace/chainN.log 2>&1 &
# Chain ids (priority order; keep all seeds of a cell on one hardware class):
#   0: Qwen3-1.7B x 100k x k8 x 3 seeds        (the interaction row - highest value)
#   1: Qwen3-1.7B x 10k  x k16 x 3 seeds, then 100k x k16 x 3 seeds
#   2: Qwen3-4B corpus prep (10k + 100k) + 40-step smoke, then 4B x 10k x k8 x 3 seeds
#   3: waits for chain 2's 100k prep marker, then 4B x 100k x k8 x 3 seeds
# Checkpoints upload to W&B as model artifacts on completion (finished work survives pod
# death); eval-final runs LOCALLY afterwards by auto-pulling those artifacts.
set -u
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CLI="uv run --group infer python -m conceptformer.cli"
CHAIN="${1:?chain id 0-3}"

# Grid-wide constants: before_entity placement, question-level split (matches every existing
# cell; strictness is applied at eval time), eff-batch 32, 72k steps, locked HPs.
snap() { case "$1" in *_10k_*) echo cftrain_10k ;; *_100k_*) echo cftrain_100k ;; esac; }

train() { # train <model> <dataset> <k> <seed> <batch> <accum> <ckpt> <group>
  $CLI cf-train --model "$1" --dataset "$2" --snapshot "$(snap "$2")" \
    --k "$3" --d-model 1024 --n-layers 4 --gate-mode none \
    --placement before_entity --split-mode question \
    --batch "$5" --grad-accum "$6" --steps 72000 --eval-every 6000 \
    --eval-n 200 --popqa-eval 200 --seed "$4" --checkpoint "$7" \
    --wandb --wandb-group "$8" --device cuda:0 || echo "FAILED $7"
}

case "$CHAIN" in
0)
  for s in 0 1 2; do
    train Qwen/Qwen3-1.7B cftrain_qa_100k_q3b17 8 "$s" 8 4 "q3b17_100k_k8_s$s" v15-scaling-17b
  done ;;
1)
  for s in 0 1 2; do
    train Qwen/Qwen3-1.7B cftrain_qa_10k_q3b17 16 "$s" 8 4 "q3b17_10k_k16_s$s" v15-scaling-17b
  done
  for s in 0 1 2; do
    train Qwen/Qwen3-1.7B cftrain_qa_100k_q3b17 16 "$s" 8 4 "q3b17_100k_k16_s$s" v15-scaling-17b
  done ;;
2)
  for c in 10k 100k; do
    mkdir -p "data/cf_train/cftrain_qa_${c}_q3b4"
    cp "data/cf_train/cftrain_qa_${c}_q3b17/qa.jsonl" \
       "data/cf_train/cftrain_qa_${c}_q3b17/manifest.json" \
       "data/cf_train/cftrain_qa_${c}_q3b4/"
  done
  $CLI tier-cftrain --dataset cftrain_qa_10k_q3b4 --snapshot cftrain_10k \
    --model Qwen/Qwen3-4B --device cuda:0 --batch-size 32
  $CLI extract-teacher-paths --dataset cftrain_qa_10k_q3b4 --snapshot cftrain_10k \
    --model Qwen/Qwen3-4B --device cuda:0 --batch-size 32
  # Memory smoke BEFORE committing days: 4B at batch 4 x accum 8 on 24 GB.
  $CLI cf-train --model Qwen/Qwen3-4B --dataset cftrain_qa_10k_q3b4 --snapshot cftrain_10k \
    --k 8 --d-model 1024 --n-layers 4 --gate-mode none --placement before_entity \
    --split-mode question --no-cache-teacher --batch 4 --grad-accum 8 --steps 40 \
    --eval-every 40 --eval-n 16 --device cuda:0 || { echo "4B SMOKE FAILED"; exit 1; }
  for s in 0 1 2; do
    train Qwen/Qwen3-4B cftrain_qa_10k_q3b4 8 "$s" 4 8 "q3b4_10k_k8_s$s" v15-scaling-4b
  done
  $CLI tier-cftrain --dataset cftrain_qa_100k_q3b4 --snapshot cftrain_100k \
    --model Qwen/Qwen3-4B --device cuda:0 --batch-size 32
  $CLI extract-teacher-paths --dataset cftrain_qa_100k_q3b4 --snapshot cftrain_100k \
    --model Qwen/Qwen3-4B --device cuda:0 --batch-size 32
  touch data/cf_train/cftrain_qa_100k_q3b4/.prep_done ;;
3)
  until [ -f data/cf_train/cftrain_qa_100k_q3b4/.prep_done ]; do sleep 300; done
  for s in 0 1 2; do
    train Qwen/Qwen3-4B cftrain_qa_100k_q3b4 8 "$s" 4 8 "q3b4_100k_k8_s$s" v15-scaling-4b
  done ;;
*) echo "unknown chain $CHAIN"; exit 1 ;;
esac
echo "=== chain $CHAIN done $(date)"
