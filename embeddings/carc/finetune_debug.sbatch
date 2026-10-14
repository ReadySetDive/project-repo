#!/bin/bash
# All three models back to back on ONE GPU in the debug partition (1 h cap, usually no queue).
# Use this when the gpu partition is queued but a debug A40 is free (check: carc/carc_gpus.ps1).
# Submit from the project-repo root on CARC, after embeddings/carc/setup.sh:
#   sbatch embeddings/carc/finetune_debug.sbatch
# Cost: 3 CPU + 16 GB + 1 A40 = 3+4+8 = 15 SU/min; ~15-25 min expected, 750 SU at the 50 min cap.
#SBATCH --account=xiangren_1987
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=sentimix-ft-debug
#SBATCH --output=embeddings/carc/logs/%x_%j.out
set -uo pipefail                                # no -e: one model failing shouldn't skip the rest

export HF_HOME=$HOME/.cache/huggingface         # all on /home1: /scratch1 hung Oct 10-12
export HF_HUB_OFFLINE=1
PY=$HOME/venvs/hing2/bin/python   # built by build_env.sh; not `uv run`: uv's cache on /scratch1 wedged Oct 10 (job 12922539)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # BLAS otherwise starts a thread per node core; on a login node that failed with pthread_create errors
export CKPT_ROOT=$HOME/sentimix-ckpt/$SLURM_JOB_ID
export KEEP_DIR=$HOME/sentimix-models/$SLURM_JOB_ID   # best model per run, kept

echo "$(date)  on $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
# smallest first, so a timeout loses the biggest model, not all of them
# override with: sbatch --export=ALL,MODELS="Qwen/Qwen3-0.6B-Base" ...
# HingmmBERT checkpoints (local paths get the 2e-5 default lr, so pass mmBERT's 5e-5):
#   C=~/hingmmbert/hingmmbert-1b; MODELS="$C/tok-100M $C/tok-200M" EXTRA="--lr 5e-5" sbatch embeddings/carc/finetune_debug.sbatch
# Extra seeds: SEEDS="1 2 3" runs each model once per seed, results in results/finetune-s<seed>/
#   (unset = the default seed 544, results in results/finetune/)
for SEED in ${SEEDS:-544}; do
    seedargs=""
    [ -n "${SEEDS:-}" ] && seedargs="--seed $SEED --tag s$SEED"
    for MODEL in ${MODELS:-l3cube-pune/hing-bert jhu-clsp/mmBERT-base codefuse-ai/F2LLM-v2-0.6B}; do
        echo "$(date)  === $MODEL  seed $SEED"
        "$PY" embeddings/finetune.py --model "$MODEL" ${EXTRA:-} $seedargs || echo "!!! $MODEL seed $SEED failed"
    done
done
echo "$(date)  done"
