#!/usr/bin/env bash
# Phase 2.4 — validate the stabilizer at the full 72k horizon (variance grows with horizon, so a
# stabilizer that helps @36k must be confirmed @72k). Base config matches rounds 1-2:
# aug-off / prefix / k8 / d1024 / L4, cached snapshot, seeded. Each arm x3 seeds.
#
# Candidate from round 2: gate-none (drop tanh gate; std 1.54->1.03pt @36k, mean held).
#   GPU0: gate-none           x3  -> confirm the stabilizer alone.
#   GPU1: gate-none + batch32 x3  -> does dropping the gate tame batch32's spread while keeping
#                                    its +6pt accuracy gain? (batch32: best mean, worst spread @36k)
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=batch  $3=tag  ${@:4}=extra flags
  local gpu=$1 batch=$2 tag=$3; shift 3
  for s in 0 1 2; do
    echo "=== P24 ${tag}_s$s START (gpu$gpu batch$batch) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
      uv run --group infer python -m conceptformer.cli cf-train \
        --dataset cftrain_qa_10k --snapshot cftrain_10k \
        --k 8 --d-model 1024 --n-layers 4 --batch $batch --steps 72000 \
        --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
        --gate-mode none \
        --wandb --wandb-group phase24-72k-validation --checkpoint p24_${tag}_s$s "$@"
    echo "=== P24 ${tag}_s$s DONE exit $? ==="
  done
}

( train 0 16 gatenone;       echo P24_GPU0_DONE ) > /tmp/cf_p24_gpu0.log 2>&1 &
( train 1 32 gatenone_b32;   echo P24_GPU1_DONE ) > /tmp/cf_p24_gpu1.log 2>&1 &
wait
echo "PHASE24_DONE"
