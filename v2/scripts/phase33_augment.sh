#!/usr/bin/env bash
# Phase 3.3 — re-test the augment ablation (F7) on the locked base with error bars. F7's "+7.7 augment
# helps generalization" DOWNGRADED: the effect flipped sign across single inits (+8.5 / -6.5), mean
# ~0 inside the noise. Re-test multi-seed.
#
# Ablation = --augment ON vs OFF at the locked config (gate-none, eff-batch-32, 72k, d1024/L4, k8).
# aug-OFF x3 = REUSE the locked config's seeds (`p25_eff32_s{0,1,2}`, group phase25-grad-accum) -- do
# NOT re-run. Only aug-ON needs training here, x3 seeds.
#
# IMPORTANT (fair comparison): aug-ON evals under a HELD-OUT prompt, aug-OFF under the training prompt,
# so cf-train's own held_out is NOT directly comparable across arms. The headline F7-style comparison
# is the 7-prompt-mean from `eval-prompt-robustness`, run on all 6 checkpoints AFTER training (the
# follow-up step, not in this script). Checkpoints are saved as model:p33_augon_s* artifacts.
#
#   GPU0: augon s0, s2.   GPU1: augon s1.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=seed
  local gpu=$1 s=$2
  echo "=== P33 augon_s$s START (gpu$gpu) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
    uv run --group infer python -m conceptformer.cli cf-train \
      --dataset cftrain_qa_10k --snapshot cftrain_10k \
      --k 8 --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 72000 \
      --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
      --gate-mode none --augment \
      --wandb --wandb-group phase33-augment --checkpoint p33_augon_s$s
  echo "=== P33 augon_s$s DONE exit $? ==="
}

( train 0 0; train 0 2; echo P33_GPU0_DONE ) > /tmp/cf_p33_gpu0.log 2>&1 &
( train 1 1;            echo P33_GPU1_DONE ) > /tmp/cf_p33_gpu1.log 2>&1 &
wait
echo "PHASE33_DONE"
