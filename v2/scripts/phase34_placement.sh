#!/usr/bin/env bash
# Phase 3.4 — placement ablation on the locked base + error bars. Where the k concept tokens sit
# relative to the entity mention may affect entity<->knowledge binding (hypothesis: adjacency /
# replacement binds tighter than a detached prefix). Prior evidence: single-seed only, untrusted.
#
# Modes: prefix (= locked config, baseline) | before_entity | after_entity | replace_entity.
# prefix x3 = REUSE `p25_eff32_s{0,1,2}` (group phase25-grad-accum); only the 3 entity-relative modes
# need training, x3 seeds each (9 runs). Locked base: gate-none, eff-batch-32, 72k, d1024/L4, k8.
# NOTE: entity-relative modes locate the entity label verbatim in the question (else fall back to
# prefix); the corpus has the label verbatim in ~100% of questions, so all modes run on full data.
#
#   GPU0: before_entity x3, then replace_entity x3.   GPU1: after_entity x3.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu  $2=mode
  local gpu=$1 mode=$2
  for s in 0 1 2; do
    echo "=== P34 ${mode}_s$s START (gpu$gpu) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$gpu \
      uv run --group infer python -m conceptformer.cli cf-train \
        --dataset cftrain_qa_10k --snapshot cftrain_10k \
        --k 8 --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 72000 \
        --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
        --gate-mode none --placement $mode \
        --wandb --wandb-group phase34-placement --checkpoint p34_${mode}_s$s
    echo "=== P34 ${mode}_s$s DONE exit $? ==="
  done
}

( train 0 before_entity; train 0 replace_entity; echo P34_GPU0_DONE ) > /tmp/cf_p34_gpu0.log 2>&1 &
( train 1 after_entity;                          echo P34_GPU1_DONE ) > /tmp/cf_p34_gpu1.log 2>&1 &
wait
echo "PHASE34_DONE"
