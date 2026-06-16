#!/usr/bin/env bash
# Phase-2 variance study (F8) — a HYPOTHESIS TEST, not just a measurement.
#
# Open questions it tests (all configs: aug-off, prefix, k=8, d1024/L4, 48k cached, NOW seeded):
#   (a) Is the run-to-run variance even driven by INIT?  -> same-seed reproducibility:
#       run seed 0 TWICE (s0 vs s0_repro).
#         * if they ~match (<~1pt): seeding controls init & CUDA noise is small -> init was the
#           uncontrolled variable behind the original 8.5pt gap.
#         * if they differ a lot: large residual (CUDA) nondeterminism -> the "init" story is wrong
#           or incomplete; seeding alone won't give reproducibility.
#   (b) What is the real seeded noise floor?  -> across-seed spread of baseline (s0,s1,s2), SAME code.
#         * if std still ~3-4pt (range ~8): variance is real & init-driven (only seed differs now).
#         * if std small: the original 8.5pt gap was mostly the old-vs-new CODE difference, not init.
#   (c) Does a larger batch reduce it?  -> batch64 across-seed std vs baseline across-seed std.
#
# GPU0: baseline batch16 {0, 0_repro, 1, 2}   GPU1: batch64 {0,1,2}   (in parallel)
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2

GPU0_LOG=/tmp/cf_aug_placement_gpu0.log
GPU1_LOG=/tmp/claude-1002/-home-joelbarmettler-projects-ConceptFormer/cf237e9c-4173-4592-9934-9f53223182e4/tasks/bndwbji7x.output

if [ "${SKIP_WAIT:-0}" != "1" ]; then
  echo "waiting for placement runs to free the GPUs..."
  while ! grep -qa "AUG_PLACEMENT_GPU0_DONE" "$GPU0_LOG" 2>/dev/null; do sleep 120; done
  while ! grep -qa "AUG_PLACEMENT_GPU1_DONE" "$GPU1_LOG" 2>/dev/null; do sleep 60; done
  echo "GPUs free; launching variance study"
  sleep 20
fi

train () {  # $1=gpu $2=batch $3=seed $4=checkpoint-tag
  echo "=== VAR $4 START (gpu$1 batch$2 seed$3) ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 \
    uv run --group infer python -m conceptformer.cli cf-train \
      --dataset cftrain_qa_10k --snapshot cftrain_10k \
      --k 8 --d-model 1024 --n-layers 4 --batch $2 --steps 48000 \
      --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $3 --device cuda:0 \
      --wandb --wandb-group variance-study --checkpoint var_$4
  echo "=== VAR $4 DONE exit $? ==="
}

# GPU0: same-seed repro pair (0 vs 0_repro) + across-seed (0,1,2)
( train 0 16 0 baseline_s0; train 0 16 0 baseline_s0repro; train 0 16 1 baseline_s1; train 0 16 2 baseline_s2; echo VAR_BASELINE_DONE ) > /tmp/cf_var_baseline.log 2>&1 &
# GPU1: batch64 across-seed (0,1,2)
( train 1 64 0 batch64_s0; train 1 64 1 batch64_s1; train 1 64 2 batch64_s2; echo VAR_BATCH64_DONE ) > /tmp/cf_var_batch64.log 2>&1 &
wait
echo "VARIANCE_STUDY_DONE"
