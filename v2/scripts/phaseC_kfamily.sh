#!/usr/bin/env bash
# Phase C — the MAIN MODEL: the k-family {1,2,4,8,16,32} trained on the 100k corpus to convergence,
# with the locked config + HPs, 1 seed each, checkpoint-SELECTED on held-out. This is the deliverable
# the token-efficiency curve (Phase D) is read from. k=8 doubles as the convergence-horizon probe
# (still climbing at 1.6 epoch in the lr check; 100k steps here = ~4 epochs to find the plateau).
#
# Locked: gate-none, eff-batch-32 (batch16/grad-accum2), before_entity, d1024/L4, --no-cache-teacher.
# HPs (Phase B): lr 1e-4, schedule constant, weight-decay 0.01, warmup-frac 0.05.
# 100k steps (~4 epochs), eval every 10k, checkpoint-select best held-out (saved as <ckpt>_best.pt +
# W&B model:<ckpt>_best artifact). 1 seed (seed 0); add 3 seeds on the headline k afterward.
#
#   GPU0: k1, k4, k16.   GPU1: k2, k8, k32.   (k doesn't change step cost materially → balanced 3/3.)
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=k
  local gpu=$1 k=$2
  echo "=== PC k${k} START (gpu$gpu) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
    uv run --group infer python -m conceptformer.cli cf-train \
      --dataset cftrain_qa_100k --snapshot cftrain_100k \
      --k $k --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 100000 \
      --eval-every 10000 --eval-n 200 --popqa-eval 200 --seed 0 --device cuda:0 \
      --gate-mode none --placement before_entity --no-cache-teacher \
      --lr 1e-4 --schedule constant --weight-decay 0.01 --warmup-frac 0.05 \
      --wandb --wandb-group phaseC-kfamily-100k --checkpoint pc_k${k}
  echo "=== PC k${k} DONE exit $? ==="
}

( train 0 1; train 0 4; train 0 16; echo PC_GPU0_DONE ) > /tmp/cf_pc_gpu0.log 2>&1 &
( train 1 2; train 1 8; train 1 32; echo PC_GPU1_DONE ) > /tmp/cf_pc_gpu1.log 2>&1 &
wait
echo "PHASEC_DONE"
