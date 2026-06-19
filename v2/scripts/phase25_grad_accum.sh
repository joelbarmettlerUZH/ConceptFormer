#!/usr/bin/env bash
# Phase 2.5 — resolve the stability/accuracy tension with gradient accumulation. The 72k validation
# (task 2.4) showed: gate-none stabilizes (std 2.05pt, acc 0.412) but batch32 is a big accuracy lever
# (acc 0.517) at a stability cost (std 3.06pt). Question: does an EVEN LARGER effective batch (via
# accumulation, which a single forward can't hold -- batch 64 OOMs) keep the accuracy gain while
# tightening spread, giving us BOTH? All arms gate-none / aug-off / prefix / k8 / d1024 / L4, x3 seeds.
#
#   GPU0: batch16 x accum4 = EFFECTIVE 64   -> the "even larger batch" test (un-runnable as a true batch).
#   GPU1: batch16 x accum2 = EFFECTIVE 32   -> control: should reproduce the true-batch32 run
#                                              (acc ~0.517, std ~3.06) up to the token-weighting approx,
#                                              validating the accumulation implementation + adding a
#                                              point to the batch-size curve {16, 32, 32-accum, 64}.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=accum  $3=tag
  local gpu=$1 accum=$2 tag=$3
  for s in 0 1 2; do
    echo "=== P25 ${tag}_s$s START (gpu$gpu batch16 accum$accum -> eff $((16*accum))) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
      uv run --group infer python -m conceptformer.cli cf-train \
        --dataset cftrain_qa_10k --snapshot cftrain_10k \
        --k 8 --d-model 1024 --n-layers 4 --batch 16 --grad-accum $accum --steps 72000 \
        --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
        --gate-mode none \
        --wandb --wandb-group phase25-grad-accum --checkpoint p25_${tag}_s$s
    echo "=== P25 ${tag}_s$s DONE exit $? ==="
  done
}

( train 0 4 eff64; echo P25_GPU0_DONE ) > /tmp/cf_p25_gpu0.log 2>&1 &
( train 1 2 eff32; echo P25_GPU1_DONE ) > /tmp/cf_p25_gpu1.log 2>&1 &
wait
echo "PHASE25_DONE"
