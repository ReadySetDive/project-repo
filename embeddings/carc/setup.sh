#!/bin/bash
# One-time setup on the CARC *login node* (compute nodes may have no internet).
# Builds the uv environment for finetune.py and downloads the three models to scratch.
# Usage (from the project-repo root on CARC):  bash embeddings/carc/setup.sh
set -euo pipefail

export HF_HOME=/scratch1/$USER/hf
export UV_CACHE_DIR=/scratch1/$USER/uv-cache     # keeps torch (~3 GB) out of the 100 GB home quota
# the CUDA wheels are 100s of MB; uv's 30 s default timed out on the login node (2026-10-10).
# Rerunning resumes: finished downloads stay in the cache.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300} UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" embeddings/carc/logs

if ! command -v uv >/dev/null; then
    echo "uv not found. Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# resolve + install the script's dependencies into the uv cache
uv run --python 3.12 embeddings/finetune.py --help >/dev/null
echo "env ok"

for m in l3cube-pune/hing-bert jhu-clsp/mmBERT-base codefuse-ai/F2LLM-v2-0.6B Qwen/Qwen3-0.6B-Base; do
    uv run --python 3.12 --with huggingface_hub hf download "$m" --quiet >/dev/null && echo "downloaded $m"
done
du -sh "$HF_HOME"
