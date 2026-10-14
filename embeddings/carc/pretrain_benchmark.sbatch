#!/bin/bash
# Speed benchmark for HingmmBERT continued pretraining: the real training code, stopped after
# 12 minutes. Prints measured tokens/s and projected hours/SU for 100M ... 2.4B tokens.
# Needs the corpus on scratch (see NOTES.md) and mmBERT-base in $HF_HOME (setup.sh).
# The corpus is unfiltered, matching HingBERT; contamination is handled at evaluation.
# Submit from the project-repo root:
#   sbatch embeddings/carc/pretrain_benchmark.sbatch
# Cost: 4 CPU + 32 GB + 1 A40 = 4+8+8 = 20 SU/min (same request as the real run, so the
# projection is in real SU); ~15-17 min elapsed, about 300-350 SU; 500 SU at the 25 min cap.
#SBATCH --account=xiangren_1987
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=hingmmbert-bench
#SBATCH --output=embeddings/carc/logs/%x_%j.out
set -euo pipefail

export HF_HOME=/scratch1/$USER/hf
export HF_HUB_OFFLINE=1
PY=$HOME/venvs/hing2/bin/python   # built by build_env.sh; not `uv run`: uv's cache on /scratch1 wedged Oct 10 (job 12922539)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # BLAS otherwise starts a thread per node core; on a login node that failed with pthread_create errors
export CKPT_ROOT=/scratch1/$USER/hingmmbert-ckpt/$SLURM_JOB_ID
export KEEP_DIR=/scratch1/$USER/hingmmbert-bench/$SLURM_JOB_ID

echo "$(date)  on $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
"$PY" embeddings/pretrain_mlm.py \
    --corpus /scratch1/$USER/hingcorpus/L3Cube-HingCorpus_roman.tar.gz \
    --benchmark-minutes 12 ${EXTRA:-}
echo "$(date)  done"
