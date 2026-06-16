#!/usr/bin/env bash
# Phase-2 variance study (F8) — a HYPOTHESIS TEST, structured so the fastest answer comes first.
#
# Tests (all: aug-off, prefix, k=8, d1024/L4, 36k cached, seeded). Read results at intermediate
# evals (every 6k) — no need to wait for the full run:
#   (a) Is it init vs CUDA?  -> var_baseline_s0 (GPU0) and var_baseline_s0repro (GPU1) run
#       CONCURRENTLY with the SAME seed. Identical per-step loss / eval => seeding controls the run
#       (init was the variable). Divergence => large residual CUDA nondeterminism. Answer in ~20min.
#   (b) Seeded noise floor   -> across-seed spread of baseline s0,s1,s2 (same code now).
#   (c) Does a larger batch reduce it -> batch32 s0,s1,s2 spread vs baseline.
#       (batch64 OOMs on a 24GB card; batch32 is the safe larger-batch test. For an even larger
#        EFFECTIVE batch without the memory cost, add gradient accumulation -- future work.)
#
# GPU0: baseline s0 -> s1 -> s2        GPU1: baseline s0repro -> batch32 s0 -> s1 -> s2
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

train () {  # $1=gpu $2=batch $3=seed $4=tag
  echo "=== VAR $4 START (gpu$1 batch$2 seed$3) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 \
    uv run --group infer python -m conceptformer.cli cf-train \
      --dataset cftrain_qa_10k --snapshot cftrain_10k \
      --k 8 --d-model 1024 --n-layers 4 --batch $2 --steps 36000 \
      --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $3 --device cuda:0 \
      --wandb --wandb-group variance-study --checkpoint var_$4
  echo "=== VAR $4 DONE exit $? ==="
}

( train 0 16 0 baseline_s0;     train 0 16 1 baseline_s1; train 0 16 2 baseline_s2; echo VAR_GPU0_DONE ) > /tmp/cf_var_gpu0.log 2>&1 &
( train 1 16 0 baseline_s0repro; train 1 32 0 batch32_s0; train 1 32 1 batch32_s1; train 1 32 2 batch32_s2; echo VAR_GPU1_DONE ) > /tmp/cf_var_gpu1.log 2>&1 &
wait
echo "VARIANCE_STUDY_DONE"
