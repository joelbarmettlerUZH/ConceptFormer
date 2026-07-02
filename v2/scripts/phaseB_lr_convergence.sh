#!/usr/bin/env bash
# Phase B convergence check — resolve lr at a REAL horizon. The 12k-step (~0.4 epoch) HP sweep ranked
# lr monotonically (5e-5 > 1e-4 > 2e-4 > 4e-4), but that is the known short-horizon bias: low lr just
# looks best because it is furthest along its stable path before higher lr converges. Re-run the lr
# contenders to ~1.6 epochs to see which actually converges best (and where held_out plateaus -> sets
# the real-run step budget). Constant schedule isolates lr (no cosine-decay confound). Other HPs from
# the sweep winner: wd 0.01, warmup-frac 0.05. Architecture = locked config, --no-cache-teacher.
#
#   GPU0: lr 5e-5, then lr 2e-4.   GPU1: lr 1e-4.   40k steps (~1.6 epoch), eval every 5k.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=lr  $3=tag
  local gpu=$1 lr=$2 tag=$3
  echo "=== LRC ${tag} START (gpu$gpu lr=$lr) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
    uv run --group infer python -m conceptformer.cli cf-train \
      --dataset cftrain_qa_100k --snapshot cftrain_100k \
      --k 8 --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 40000 \
      --eval-every 5000 --eval-n 200 --popqa-eval 0 --seed 0 --device cuda:0 \
      --gate-mode none --placement before_entity --no-cache-teacher \
      --lr $lr --schedule constant --warmup-frac 0.05 --weight-decay 0.01 \
      --wandb --wandb-group phaseB-lr-convergence --checkpoint lrc_${tag}
  echo "=== LRC ${tag} DONE exit $? ==="
}

( train 0 5e-5 lr5e5; train 0 2e-4 lr2e4; echo LRC_GPU0_DONE ) > /tmp/cf_lrc_gpu0.log 2>&1 &
( train 1 1e-4 lr1e4;                      echo LRC_GPU1_DONE ) > /tmp/cf_lrc_gpu1.log 2>&1 &
wait
echo "LR_CONVERGENCE_DONE"
