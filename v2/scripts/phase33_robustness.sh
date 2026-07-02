#!/usr/bin/env bash
# Phase 3.3 follow-up — the FAIR augment comparison: 7-prompt robustness mean per checkpoint
# (aug-on evals under a held-out prompt, aug-off under the training prompt, so cf-train's own
# held_out is not cross-arm comparable; this harness scores BOTH arms under the SAME 7 prompts).
# Runs on GPU1 while GPU0 finishes aug-on s2. p33_augon_s2 is appended once it lands (see ARGS).
#
# Usage: bash scripts/phase33_robustness.sh "<ckpt1> <ckpt2> ...".  Each result -> /tmp/cf_p33rob_<ckpt>.log
set -u
cd /home/joelbarmettler/projects/ConceptFormer/v2
CKPTS=${1:-"p25_eff32_s0 p25_eff32_s1 p25_eff32_s2 p33_augon_s0 p33_augon_s1"}

for ck in $CKPTS; do
  echo "=== ROB $ck START ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 \
    uv run --group infer python -m conceptformer.cli eval-prompt-robustness \
      --checkpoint $ck --dataset cftrain_qa_10k --snapshot cftrain_10k \
      --eval-n 200 --popqa-n 200 --seed 0 --device cuda:0 \
      > /tmp/cf_p33rob_$ck.log 2>&1
  echo "=== ROB $ck DONE exit $? ==="
done
echo "P33_ROBUSTNESS_DONE"
