#!/usr/bin/env bash
# Phase 3.1 — re-run the capacity sweep (F1) with the LOCKED stabilized config + error bars. F1's
# original claim ("bigger encoder is not better; 231M <= 21M") came from single-seed, 24k, subsample
# runs whose 8-pt spread sat inside the ~8-pt init noise (DOWNGRADED). Re-test with the locked base:
#   gate-none, batch16 x grad-accum2 (effective 32), 72k, k8, cached teacher (no subsample), x3 seeds.
# Noise floor is now ~1pt (eff32-accum std 1.08), so a real saturation/decline becomes detectable.
#
# Grid spans the 23x param range from the original sweep (d_model x n_layers):
#   d512/L2 (~10M)  d768/L2 (~21M, old sweet spot)  [d1024/L4 ~70M = REUSE p25_eff32 0.535+/-1.08]
#   d1536/L6 (~231M, the big one F1 said loses).
# The d1024/L4 point is the locked config itself -> reuse group `phase25-grad-accum` (p25_eff32_s*),
# do NOT re-run it here.
#
#   GPU0: d1536/L6 x3  (the slow 231M arm).      GPU1: d512/L2 x3 then d768/L2 x3 (fast, small).
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=d_model  $3=n_layers
  local gpu=$1 d=$2 L=$3
  for s in 0 1 2; do
    echo "=== P31 d${d}_L${L}_s$s START (gpu$gpu) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
      uv run --group infer python -m conceptformer.cli cf-train \
        --dataset cftrain_qa_10k --snapshot cftrain_10k \
        --k 8 --d-model $d --n-layers $L --batch 16 --grad-accum 2 --steps 72000 \
        --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
        --gate-mode none \
        --wandb --wandb-group phase31-capacity --checkpoint p31_d${d}_L${L}_s$s
    echo "=== P31 d${d}_L${L}_s$s DONE exit $? ==="
  done
}

( train 0 1536 6; echo P31_GPU0_DONE ) > /tmp/cf_p31_gpu0.log 2>&1 &
( train 1 512 2; train 1 768 2; echo P31_GPU1_DONE ) > /tmp/cf_p31_gpu1.log 2>&1 &
wait
echo "PHASE31_DONE"
