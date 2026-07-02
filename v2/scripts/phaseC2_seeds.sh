#!/usr/bin/env bash
# Phase C2 — error bars on the WHOLE 100k k-family (user: seed all k, not just the headline; 1 GPU;
# fine to run for days). Phase C gave seed 0 (pc_k{k}); add seeds 1 and 2 → 3 per k. Same locked
# config + HPs + 100k-step horizon + best-held-out checkpoint. GPU0 only (GPU1 left free for the
# capability experiments). Order front-loads the headline/knee k (8,16,4) so their error bars land first.
# ~12 runs x ~5.5h ≈ 3 days.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=k  $2=seed
  local k=$1 s=$2
  echo "=== C2 k${k}_s${s} START $(date -u +%FT%TZ) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
    uv run --group infer python -m conceptformer.cli cf-train \
      --dataset cftrain_qa_100k --snapshot cftrain_100k \
      --k $k --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 100000 \
      --eval-every 10000 --eval-n 200 --popqa-eval 200 --seed $s --device cuda:0 \
      --gate-mode none --placement before_entity --no-cache-teacher \
      --lr 1e-4 --schedule constant --weight-decay 0.01 --warmup-frac 0.05 \
      --wandb --wandb-group phaseC-kfamily-100k --checkpoint pc_k${k}_s${s}
  echo "=== C2 k${k}_s${s} DONE exit $? ==="
}

for k in 8 16 4 32 2 1; do
  for s in 1 2; do train $k $s; done
done
echo "PHASEC2_SEEDS_DONE"
