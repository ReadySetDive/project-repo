#!/bin/bash
# HingmmBERT: continued MLM pretraining of mmBERT-base on 1B tokens of HingCorpus
# (unfiltered, like HingBERT), lr 5e-5, 15% masking, batch 64 x 512, checkpoints every 100M tokens.
# Each checkpoint is a full model folder that finetune.py can load with --model <path>.
#
# Runs in 2 h chunks, because short jobs start far sooner than long ones (Oct 12: 1-2 h jobs
# started within minutes while 8-12 h jobs waited 12+ h). Each chunk trains for 100 min, saves a
# resumable checkpoint (weights + optimizer + LR schedule + corpus position), and, if the run
# isn't finished, submits the next chunk. --dependency=singleton keeps chunks of the same
# job name strictly one at a time, so a duplicate submission waits instead of training twice.
#
# Start (or restart after a failure) from the project-repo root:
#   sbatch embeddings/carc/pretrain_pilot.sbatch
# Same command resumes: it picks up the latest checkpoint for this job name. A chunk that crashed,
# hung (watchdog) or hit the time limit does NOT resubmit; fix the cause and run the command again.
# Stop the chain: scancel --me --name=hingmmbert-1b
#
# Files (all on /home1; /scratch1 hung on Oct 10-11):
#   ~/hingmmbert/hingmmbert-1b/tok-*        model checkpoints (100M ... 1000M), DONE when finished
#   ~/hingmmbert-ckpt/hingmmbert-1b/        resumable trainer checkpoint (~4 GB, latest only)
#   embeddings/carc/logs/hingmmbert-1b_<jobid>.out, and chunks.txt in the run folder
#
# Speeds measured: A40 26.4k tokens/s (1B = 10.5 h, ~7 chunks), L40S 44.7k (job 12930285).
# No L40S: all of them were stuck or drained on Oct 12. No V100s: our PyTorch build lacks sm_70.
# Cost: 20 SU/min, 2.4k SU per chunk.
#
# Runs a fixed venv's python on /home1, not `uv run`: job 12922539 sat 6 h at 0% CPU because uv's
# cache on /scratch1 wedged. Build the venv once: bash embeddings/carc/build_env.sh all
# If training hasn't logged a step within 15 min of start, the job cancels itself.
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --constraint=a40|a100
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=hingmmbert-1b
#SBATCH --dependency=singleton
#SBATCH --output=embeddings/carc/logs/%x_%j.out
set -euo pipefail

export HF_HOME=$HOME/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # BLAS otherwise starts a thread per node core
RUN=$SLURM_JOB_NAME
export KEEP_DIR=$HOME/hingmmbert                          # model checkpoints: $KEEP_DIR/$RUN/tok-*
export CKPT_ROOT=$HOME/hingmmbert-ckpt/$RUN               # resumable trainer state, shared by all chunks
MODEL=$HOME/models/mmBERT-base                            # hf download jhu-clsp/mmBERT-base --local-dir ~/models/mmBERT-base
CORPUS=$HOME/data/L3Cube-HingCorpus_roman.tar.gz

PY=$HOME/venvs/hing2/bin/python
LOG=embeddings/carc/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out
[ -x "$PY" ] || { echo "no venv at $PY: build it first (commands at the top of this file)"; exit 1; }
[ -f "$MODEL/config.json" ] || { echo "no model at $MODEL"; exit 1; }
[ -f "$CORPUS" ] || { echo "no corpus at $CORPUS"; exit 1; }
[ -f "$KEEP_DIR/$RUN/DONE" ] && { echo "$RUN already finished ($KEEP_DIR/$RUN/DONE)"; exit 0; }

mkdir -p "$KEEP_DIR/$RUN"
echo "$SLURM_JOB_ID $(date +%FT%T) $(hostname)" >> "$KEEP_DIR/$RUN/chunks.txt"
echo "$(date)  chunk $(wc -l < "$KEEP_DIR/$RUN/chunks.txt") on $(hostname)  $(nvidia-smi --query-gpu=name --format=csv,noheader)"
"$PY" embeddings/pretrain_mlm.py --model "$MODEL" --corpus "$CORPUS" --run "$RUN" \
    --tokens 1B --save-at 100M,200M,300M,400M,500M,600M,700M,800M,900M,1B --lr 5e-5 \
    --stop-after-minutes 100 "$@" &
pid=$!

# startup watchdog: a hang anywhere before training is running (python start, model load, corpus
# skip on resume) ends the job instead of burning the time limit. The first progress line is
# {"step": N, ...} a couple of minutes in. Not anchored with ^: tqdm redraws with \r and no
# newline, so the JSON lands mid-line after the progress bar (job 12930285 was killed while training).
for _ in $(seq 90); do
    grep -q '{"step"' "$LOG" && break
    kill -0 $pid 2>/dev/null || break      # python already exited (wait below reports how)
    sleep 10
done
if ! grep -q '{"step"' "$LOG" && kill -0 $pid 2>/dev/null; then
    echo "$(date)  no training progress in 15 min, cancelling (hung on startup?)"
    scancel "$SLURM_JOB_ID"                # scancel, not just kill: a process stuck on the filesystem ignores signals
fi
if wait $pid; then status=0; else status=$?; fi
echo "$(date)  python exited with status $status"
ls -la "$KEEP_DIR/$RUN"/

# Next chunk: only after a clean stop (status 0) of an unfinished run, and at most MAX_CHUNKS
# (default 15) chunks in all. It keeps this job's name, time limit and GPU constraint, so a
# submission like `sbatch --job-name=X --time=Y ...` chains as X/Y too.
MAX_CHUNKS=${MAX_CHUNKS:-15}
if [ "$status" -eq 0 ] && [ ! -f "$KEEP_DIR/$RUN/DONE" ]; then
    if [ "$(wc -l < "$KEEP_DIR/$RUN/chunks.txt")" -lt "$MAX_CHUNKS" ]; then
        read -r limit feat < <(squeue -h -j "$SLURM_JOB_ID" -o "%l %f")
        sbatch --job-name="$SLURM_JOB_NAME" --time="$limit" --constraint="$feat" \
            embeddings/carc/pretrain_pilot.sbatch "$@"
    else
        echo "$MAX_CHUNKS chunks used and still not done; not resubmitting"
    fi
fi
exit "$status"
