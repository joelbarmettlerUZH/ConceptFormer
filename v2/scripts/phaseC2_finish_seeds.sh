#!/usr/bin/env bash
# Phase C2 — finish the 5 MISSING 100k k-family seeds so the curve is uniformly 3-seed.
# Already done: k4/k8/k16 ×3, k32 ×2 (seed0,s1), k1/k2 ×1 (seed0). Missing exactly:
#   k1_s1, k1_s2, k2_s1, k2_s2, k32_s2.
# Locked config + HPs (lr1e-4/constant/wd0.01/warmup0.05), 100k steps, best-ckpt, group
# phaseC-kfamily-100k. GPU0: k1_s1, k2_s1, k32_s2 (3). GPU1: k1_s2, k2_s2 (2). ~10.5h/run.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=k  $3=seed
  local gpu=$1 k=$2 s=$3
  echo "=== FIN k${k}_s${s} START $(date -u +%FT%TZ) (gpu$gpu) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
    uv run --group infer python -m conceptformer.cli cf-train \
      --dataset cftrain_qa_100k --snapshot cftrain_100k \
      --k $k --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 100000 \
      --eval-every 10000 --eval-n 200 --popqa-eval 200 --seed $s --device cuda:0 \
      --gate-mode none --placement before_entity --no-cache-teacher \
      --lr 1e-4 --schedule constant --weight-decay 0.01 --warmup-frac 0.05 \
      --wandb --wandb-group phaseC-kfamily-100k --checkpoint pc_k${k}_s${s}
  echo "=== FIN k${k}_s${s} DONE exit $? ==="
}

( train 0 1 1; train 0 2 1; train 0 32 2; echo FIN_GPU0_DONE ) > /tmp/cf_finseeds_gpu0.log 2>&1 &
( train 1 1 2; train 1 2 2;              echo FIN_GPU1_DONE ) > /tmp/cf_finseeds_gpu1.log 2>&1 &
wait
echo "FINISH_SEEDS_DONE"
