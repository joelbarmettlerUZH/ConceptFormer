#!/usr/bin/env bash
# Post-review reruns (2026-07-28). Two phases, two GPU lanes, run detached.
#   Phase A (cheap, eval-only, ~6h): (1) cross-lingual 3-seed for 0.6B -- eval the s1/s2
#     checkpoints on the EN-label and DE-label(fullde,sys-de) conditions so S5.6 gets error bars;
#     (2) naive-label MetaQA transfer -- re-run transfer with underscore->space relation labels
#     (snapshot metaqa_naive) to show transfer is not an artifact of the hand-written label map.
#   Phase B (~2 days): retrain the before_entity 10k k-curve (k in {1,2,4,16,32} x 3 seeds; k=8
#     already exists as p34_before_entity) + eval-final each, so fig_kcurves and tab:grid share the
#     before_entity placement and the 10k->100k substitution ratio is single-placement.
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QA_DE=data/analysis/multilingual/popqa_de.jsonl
QA_MQ=data/snapshots/metaqa/qa_test.txt

xling () {  # $1=k  $2=seed-suffix (s1|s2)
  local k=$1 s=$2 ck=pc_k$1_$2_best
  uv run --group infer python -m conceptformer.cli eval-multilingual \
    --checkpoint $ck --qa $QA_DE --snapshot popqa_full \
    --benchmark popqa_de_0.6b_k$k --mention en --system-lang en \
    --model Qwen/Qwen3-0.6B --device cuda:0 2>&1 | grep -aE "concept=|error|Traceback" || echo "FAIL xling-en k$k $s"
  uv run --group infer python -m conceptformer.cli eval-multilingual \
    --checkpoint $ck --qa $QA_DE --snapshot popqa_full_de \
    --benchmark popqa_fullde_0.6b_k$k --mention localized --system-lang de \
    --model Qwen/Qwen3-0.6B --device cuda:0 2>&1 | grep -aE "concept=|error|Traceback" || echo "FAIL xling-de k$k $s"
}

naive () {  # $1=k  $2=ckpt-name
  uv run --group infer python -m conceptformer.cli eval-transfer \
    --checkpoint $2 --qa $QA_MQ --snapshot metaqa_naive --benchmark metaqa_1hop_naive \
    --source metaqa --n 0 --gen-batch 64 --model Qwen/Qwen3-0.6B --device cuda:0 \
    2>&1 | grep -aE "concept=|error|Traceback" || echo "FAIL naive $2"
}

trainq () {  # $1=k  $2=seed
  local k=$1 s=$2
  echo "=== TRAIN be10k_k${k}_s$s START ==="
  uv run --group infer python -m conceptformer.cli cf-train \
    --dataset cftrain_qa_10k --snapshot cftrain_10k \
    --k $k --d-model 1024 --n-layers 4 --batch 16 --grad-accum 2 --steps 72000 \
    --eval-every 6000 --eval-n 200 --popqa-eval 0 --seed $s --device cuda:0 \
    --gate-mode none --placement before_entity \
    --wandb --wandb-group kcurve-before-entity-10k --checkpoint be10k_k${k}_s$s \
    2>&1 | tail -2
  echo "=== EVAL-FINAL be10k_k${k}_s$s ==="
  uv run --group infer python -m conceptformer.cli eval-final \
    --checkpoint be10k_k${k}_s$s --dataset cftrain_qa_10k --snapshot cftrain_10k \
    --popqa-snapshot popqa_full --device cuda:0 2>&1 | grep -aE "popqa|held|error|Traceback" | tail -4
  echo "=== be10k_k${k}_s$s DONE ==="
}

lane () {  # $1=gpu ; runs its assigned Phase A then Phase B share (k-list passed via $2.. )
  local gpu=$1; shift
  export CUDA_VISIBLE_DEVICES=$gpu
  for k in "$@"; do for s in s1 s2; do xling $k $s; done; done
  for k in "$@"; do for c in best s1_best s2_best; do naive $k pc_k${k}_${c}; done; done
  echo "### PHASE_A_LANE${gpu}_DONE"
}

# ---- Phase A: split k across the two GPUs ----
( lane 0 1 2 4 ; echo "### LANE0_A_DONE" ) > /tmp/rerun_gpu0.log 2>&1 &
( lane 1 8 16 32 ; echo "### LANE1_A_DONE" ) > /tmp/rerun_gpu1.log 2>&1 &
wait
echo "### PHASE_A_DONE $(date +%H:%M)" | tee -a /tmp/rerun_gpu0.log

# ---- Phase B: retrain before_entity 10k k-curve (k=8 already exists) ----
( export CUDA_VISIBLE_DEVICES=0; for k in 1 2 4;  do for s in 0 1 2; do trainq $k $s; done; done; echo "### LANE0_B_DONE" ) >> /tmp/rerun_gpu0.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=1; for k in 16 32;  do for s in 0 1 2; do trainq $k $s; done; done; echo "### LANE1_B_DONE" ) >> /tmp/rerun_gpu1.log 2>&1 &
wait
echo "### ALL_DONE $(date +%H:%M)" | tee -a /tmp/rerun_gpu0.log
