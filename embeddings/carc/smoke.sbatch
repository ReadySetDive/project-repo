#!/bin/bash
# Smoke test for the env and the env bundle, on one debug GPU. Each step prints PASS/FAIL and the
# job carries on, so one run shows everything that's broken.
#   1 cuda      ~/venvs/hing2 sees the GPU and does a bf16 matmul
#   2 pretrain  real pretrain_mlm.py for 3M tokens (~90 steps): mmBERT + corpus off /scratch1,
#               the {"step": 50} line the pilot's watchdog waits for, a checkpoint save
#   3 finetune  finetune.py on that checkpoint, 300 examples, 1 epoch (the checkpoint-eval path)
#   4 bundle    rebuild the env from ~/hing-env-bundle.tar on node-local disk, offline, then
#               re-run step 1 with it (proves the artifact works; skipped if no bundle)
# Submit from the project-repo root (after `build_env.sh bundle` if you want step 4):
#   sbatch embeddings/carc/smoke.sbatch
# Cost: 20 SU/min (4 CPU + 32 GB + 1 A40), ~25 min with the bundle step, ~500 SU.
#SBATCH --account=xiangren_1987
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=a40:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=smoke
#SBATCH --output=embeddings/carc/logs/%x_%j.out
set -uo pipefail                                   # no -e: a failed step is reported, not fatal

export HF_HOME=/scratch1/$USER/hf
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export CKPT_ROOT=/scratch1/$USER/smoke-ckpt/$SLURM_JOB_ID
PY=$HOME/venvs/hing2/bin/python
BUNDLE=${BUNDLE:-$HOME/hing-env-bundle.tar}
OUT=/scratch1/$USER/smoke/$SLURM_JOB_ID
LOCAL=${TMPDIR:-/tmp}/hing-smoke-$SLURM_JOB_ID          # node-local disk for the bundle rebuild
declare -A RESULT

step() {                                           # step NAME MINUTES command...
    local name=$1 mins=$2; shift 2
    echo; echo "===== $name  $(date +%T)"
    local t0=$SECONDS
    if timeout "${mins}m" "$@"; then RESULT[$name]="PASS ($((SECONDS - t0)) s)"
    else RESULT[$name]="FAIL rc=$? ($((SECONDS - t0)) s)"; fi
    echo "===== $name: ${RESULT[$name]}"
}

CUDA_CHECK='import torch, transformers
assert torch.cuda.is_available(), "no cuda"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); y = (x @ x).float().sum().item()
print("cuda ok:", torch.cuda.get_device_name(), "torch", torch.__version__, "transformers", transformers.__version__)'

echo "$(date)  on $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"

step cuda 3 "$PY" -c "$CUDA_CHECK"

step pretrain 12 env KEEP_DIR="$OUT" "$PY" embeddings/pretrain_mlm.py \
    --corpus /scratch1/$USER/hingcorpus/L3Cube-HingCorpus_roman.tar.gz \
    --tokens 3M --save-at 3M --run smoke
[ -f "$OUT/smoke/tok-3M/config.json" ] || RESULT[pretrain]="FAIL (no checkpoint at $OUT/smoke/tok-3M)"

step finetune 12 env -u KEEP_DIR "$PY" embeddings/finetune.py \
    --model "$OUT/smoke/tok-3M" --lr 5e-5 --limit 300 --epochs 1 --tag smoke

if [ -f "$BUNDLE" ]; then
    mkdir -p "$LOCAL"
    step bundle-unpack 5 tar -xf "$BUNDLE" -C "$LOCAL"
    step bundle-install 15 env PYTHON="$LOCAL/python/bin/python3.12" WHEELS="$LOCAL/wheels" ENV="$LOCAL/env" \
        LOCK="$LOCAL/requirements-lock.txt" bash "$LOCAL/build_env.sh" install
    step bundle-cuda 3 "$LOCAL/env/bin/python" -c "$CUDA_CHECK"
    rm -rf "$LOCAL"
else
    RESULT[bundle]="SKIPPED (no $BUNDLE; run build_env.sh bundle first)"
fi

rm -rf "$OUT" "$CKPT_ROOT"                          # smoke checkpoints aren't worth keeping
echo; echo "===== SUMMARY  $(date)"
for k in cuda pretrain finetune bundle bundle-unpack bundle-install bundle-cuda; do
    [ -n "${RESULT[$k]:-}" ] && printf '  %-15s %s\n' "$k" "${RESULT[$k]}"
done
