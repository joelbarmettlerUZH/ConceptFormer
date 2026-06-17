#!/usr/bin/env bash
# Phase-2 Round 2 — stabilization arms (test several, pick winner). 36k (per decision to stay at 36k;
# the WINNER is later validated at full 72k, task 2.4). Same base config as round 1
# (aug-off/prefix/k8/d1024/L4, cached, seeded). Each arm x3 seeds; compare across-seed std/range to
# the baseline (mean .330, std 1.5pt, range 3.5pt held_out @36k).
#
# Arms:  EMA (--ema-decay 0.999) | gate-none (--gate-mode none) | gentler-opt (--grad-clip 1.0)
# GPU0 (free now): ema {0,1,2} then gate-none {0,1,2}.  GPU1 (after batch32 frees it): gentler {0,1,2}.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2
GPU1_BATCH32_LOG=/tmp/cf_var_batch32.log

train () {  # $1=gpu  $2=tag  ${@:3}=extra cf-train flags
  local gpu=$1 tag=$2; shift 2
  for s in 0 1 2; do
    echo "=== R2 ${tag}_s$s START (gpu$gpu) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
      uv run --group infer python -m conceptformer.cli cf-train \
        --dataset cftrain_qa_10k --snapshot cftrain_10k \
        --k 8 --d-model 1024 --n-layers 4 --batch 16 --steps 36000 \
        --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
        --wandb --wandb-group round2-stabilization --checkpoint r2_${tag}_s$s "$@"
    echo "=== R2 ${tag}_s$s DONE exit $? ==="
  done
}

( train 0 ema --ema-decay 0.999; train 0 gatenone --gate-mode none; echo R2_GPU0_DONE ) > /tmp/cf_r2_gpu0.log 2>&1 &
(
  while ! grep -qa "VAR_BATCH32_DONE" "$GPU1_BATCH32_LOG" 2>/dev/null; do sleep 60; done
  train 1 gradclip --grad-clip 1.0
  echo R2_GPU1_DONE
) > /tmp/cf_r2_gpu1.log 2>&1 &
wait
echo "ROUND2_DONE"
