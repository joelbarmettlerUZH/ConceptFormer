#!/usr/bin/env bash
# Bootstrap a fresh RunPod (or any Ubuntu CUDA box) for ConceptFormer v1.5 training chains.
#
# Required env:
#   HF_TOKEN     - HuggingFace token (read access to joelbarmettler/conceptformer-data)
#   WANDB_API_KEY - W&B key for entity university-of-zurich (checkpoints upload as artifacts,
#                   which is how finished work is banked off-pod; a dead pod loses only the
#                   in-flight runs, never completed ones)
# Usage:  bash pod_bootstrap.sh   (idempotent; safe to re-run)
#
# After bootstrap, launch one sequential chain per GPU, e.g.:
#   cd ~/ConceptFormer/v2 && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/pod_chain_example.sh > /workspace/chain0.log 2>&1 &
# (CUDA_VISIBLE_DEVICES=N + --device cuda:0 — never --device cuda:N.)
set -euo pipefail

echo "=== [1/5] system deps"
command -v git >/dev/null || (apt-get update && apt-get install -y git)
command -v curl >/dev/null || apt-get install -y curl

echo "=== [2/5] uv + repo"
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d "$HOME/ConceptFormer" ]; then
  git clone --branch conceptformer-v2 --depth 1 \
    https://github.com/joelbarmettlerUZH/ConceptFormer.git "$HOME/ConceptFormer"
fi
cd "$HOME/ConceptFormer/v2"
uv sync --group infer

echo "=== [3/5] auth"
: "${HF_TOKEN:?set HF_TOKEN}"
: "${WANDB_API_KEY:?set WANDB_API_KEY}"
uv run hf auth login --token "$HF_TOKEN" >/dev/null 2>&1 || hf auth login --token "$HF_TOKEN"

echo "=== [4/5] data (snapshots + prepped corpora from the HF dataset repo)"
# data/ is git-ignored and machine-local by design. The HF repo uses explicit teacher names
# (reader-facing); the pipeline's internal directory names are shorter -- map them here.
uv run hf download joelbarmettler/conceptformer-data --repo-type dataset --local-dir data
declare -A NAME_MAP=(
  ["cftrain_qa_10k_qwen3-0.6b"]="cftrain_qa_10k"
  ["cftrain_qa_100k_qwen3-0.6b"]="cftrain_qa_100k"
  ["cftrain_qa_10k_qwen3-1.7b"]="cftrain_qa_10k_q3b17"
  ["cftrain_qa_100k_qwen3-1.7b"]="cftrain_qa_100k_q3b17"
  ["cftrain_qa_10k_qwen3.5-0.8b"]="cftrain_qa_10k_q35b08"
  ["cftrain_qa_10k_qwen3.5-2b"]="cftrain_qa_10k_q35b2"
)
for hf_name in "${!NAME_MAP[@]}"; do
  src="data/cf_train/$hf_name"; dst="data/cf_train/${NAME_MAP[$hf_name]}"
  [ -d "$src" ] && [ ! -d "$dst" ] && mv "$src" "$dst"
done

echo "=== [5/5] smoke"
uv run python -c "
from conceptformer.config import settings
from conceptformer.data.snapshot import iter_subgraphs
n = sum(1 for _ in zip(range(3), iter_subgraphs(settings.snapshots_dir / 'cftrain_10k')))
assert n == 3, 'snapshot read failed'
print('data OK; GPUs:')
"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "=== bootstrap complete. Launch one chain per GPU (see header)."
