#!/usr/bin/env bash
# Phase 3.2 — re-run the k-curve (F2) at the best capacity from 3.1 (d1024/L4, ~70M) with the LOCKED
# base + error bars. F2's original claim (starved at k=1-2, jumps at k=4, peaks at k=8, flat after;
# knee k~4-8) was single-seed @24k subsample, inside the ~8-pt noise (DOWNGRADED). This is the paper's
# token-efficiency headline (k soft tokens standing in for ~100-200 fact tokens), so run a full curve.
#
# Locked base: gate-none, batch16 x grad-accum2 (eff 32), 72k, d1024/L4, cached teacher, x3 seeds.
# Grid k in {1,2,4,8,16}: k8 = the locked config itself -> REUSE p25_eff32_* (0.535+/-1.08); do NOT
# re-run it. New points k in {1,2,4,16}, balanced 6/6 across the two GPUs.
#
#   GPU0: k1 x3 then k16 x3.     GPU1: k2 x3 then k4 x3.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=k
  local gpu=$1 k=$2
  for s in 0 1 2; do
    echo "=== P32 k${k}_s$s START (gpu$gpu) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
      uv run --group infer python -m conceptformer.cli cf-train \
        --dataset cftrain_qa_10k --snapshot cftrain_10k \
        --k $k --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 72000 \
        --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
        --gate-mode none \
        --wandb --wandb-group phase32-kcurve --checkpoint p32_k${k}_s$s
    echo "=== P32 k${k}_s$s DONE exit $? ==="
  done
}

( train 0 1; train 0 16; echo P32_GPU0_DONE ) > /tmp/cf_p32_gpu0.log 2>&1 &
( train 1 2; train 1 4;  echo P32_GPU1_DONE ) > /tmp/cf_p32_gpu1.log 2>&1 &
wait
echo "PHASE32_DONE"
