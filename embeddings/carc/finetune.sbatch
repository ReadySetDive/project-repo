#!/bin/bash
# Fine-tune the three models on SentiMix, one GPU each (job array index -> model).
# Submit from the project-repo root on CARC, after embeddings/carc/setup.sh:
#   sbatch embeddings/carc/finetune.sbatch
#   sbatch --array=2 embeddings/carc/finetune.sbatch          only F2LLM
#   sbatch --export=ALL,EXTRA="--seed 1 --tag s1" embeddings/carc/finetune.sbatch
# Cost: 1 A40 + 4 CPU + 32 GB = 8+4+8 = 20 SU/min per job, so ~15 min x 3 jobs is about 900 SU.
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --job-name=sentimix-ft
#SBATCH --output=embeddings/carc/logs/%x_%A_%a.out
set -euo pipefail

MODELS=(l3cube-pune/hing-bert jhu-clsp/mmBERT-base codefuse-ai/F2LLM-v2-0.6B)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

export HF_HOME=/scratch1/$USER/hf
export HF_HUB_OFFLINE=1          # everything was fetched by setup.sh
PY=$HOME/venvs/hing2/bin/python   # built by build_env.sh; not `uv run`: uv's cache on /scratch1 wedged Oct 10 (job 12922539)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # BLAS otherwise starts a thread per node core; on a login node that failed with pthread_create errors
export CKPT_ROOT=/scratch1/$USER/sentimix-ckpt/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export KEEP_DIR=/scratch1/$USER/sentimix-models/${SLURM_ARRAY_JOB_ID}   # best model per run, kept

echo "$(date)  $MODEL on $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
"$PY" embeddings/finetune.py --model "$MODEL" ${EXTRA:-}
echo "$(date)  done"
