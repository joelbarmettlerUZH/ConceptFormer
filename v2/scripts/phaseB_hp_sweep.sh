#!/usr/bin/env bash
# Phase B — training-HP sweep on the 100k corpus (architecture FIXED to the locked config per the
# "HP only, keep d1024/L4" decision). W&B Bayesian sweep, one agent per GPU, reduced budget (~0.4
# epoch) just to RANK lr/schedule/warmup/weight-decay before the real converged run.
#
# Locked/fixed (passed as single-value sweep params so cf-sweep pins them): gate-mode none,
# grad-accum 2 (eff batch 32 with --batch 16), placement before_entity, d-model 1024, n-layers 4, k 8,
# --no-cache-teacher (925k rows -> ~78GB if cached; live teacher instead).
# Swept: lr {5e-5,1e-4,2e-4,4e-4} x schedule {cosine,constant} x warmup-frac {0.02,0.05}
#        x weight-decay {0.0,0.01}.  Metric: held_out/concept_acc (maximize).
#
# 1 epoch @ eff-batch-32 = ~28,900 steps; 12000 steps ~= 0.42 epoch, enough to rank HPs.
set -eu
cd /home/joelbarmettler/projects/ConceptFormer/v2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run --group infer python -m conceptformer.cli cf-sweep \
  --params "gate-mode=none;grad-accum=2;placement=before_entity;d-model=1024;n-layers=4;k=8;lr=5e-5,1e-4,2e-4,4e-4;schedule=cosine,constant;warmup-frac=0.02,0.05;weight-decay=0.0,0.01" \
  --dataset cftrain_qa_100k --snapshot cftrain_100k \
  --steps 12000 --batch 16 --eval-n 200 --eval-every 2000 --popqa-eval 0 \
  --no-subsample --no-cache-teacher \
  --method bayes --count 9 --devices cuda:0,cuda:1 --name hp-sweep-100k
