#!/bin/bash
# Micro-batch sweep for HingmmBERT pretraining: the real training code (benchmark mode, no
# checkpoints) at micro-batch 8/16/32/64, 3 minutes each, on one GPU. The optimizer batch stays
# 64 sequences (HingBERT's recipe); only how it's split into forward passes changes. An OOM just
# marks that size as too big. Prints a tokens/s and peak-memory table at the end.
#
# Submit from the project-repo root, once per GPU type:
#   sbatch --constraint=a40 embeddings/carc/microbatch_sweep.sbatch
#   sbatch --constraint=a100 embeddings/carc/microbatch_sweep.sbatch
# Sizes to try: MBS="16 32" sbatch ...   Cost: 20 SU/min, ~20 min = ~400 SU per GPU type.
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a40|a100
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --job-name=hingmmbert-mbsweep
#SBATCH --output=embeddings/carc/logs/%x_%j.out
set -uo pipefail                                           # no -e: one OOM shouldn't end the sweep

export HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export KEEP_DIR=$HOME/hingmmbert-bench/$SLURM_JOB_ID CKPT_ROOT=$HOME/hingmmbert-bench/$SLURM_JOB_ID/trainer
PY=$HOME/venvs/hing2/bin/python
MODEL=$HOME/models/mmBERT-base
CORPUS=$HOME/data/L3Cube-HingCorpus_roman.tar.gz
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo "$(date)  on $(hostname)  $GPU"

summary=""
for mb in ${MBS:-8 16 32 64}; do
    echo; echo "=== micro-batch $mb ==="
    out=$("$PY" embeddings/pretrain_mlm.py --model "$MODEL" --corpus "$CORPUS" --micro-batch "$mb" \
          --benchmark-minutes 3 2>&1 | tr '\r' '\n' | grep -v '%|' | tee /dev/stderr)
    line=$(grep '^== benchmark' <<<"$out")
    if [ -n "$line" ]; then
        summary+="micro-batch $mb: ${line#== benchmark: }"$'\n'
    elif grep -qi 'out of memory' <<<"$out"; then
        summary+="micro-batch $mb: out of memory"$'\n'
    else
        summary+="micro-batch $mb: failed (see above)"$'\n'
    fi
done
rm -rf "$KEEP_DIR"
echo; echo "=== SUMMARY  $GPU ==="; printf '%s' "$summary"
