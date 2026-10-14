#!/bin/bash
# Sync this repo with ~/project-repo on CARC. Only changed files are sent.
#
#   push   send code to CARC: embeddings/*.py, embeddings/carc/ (minus logs), SentiMix data
#   pull   bring back embeddings/results/ and the Slurm logs
#
# Run from Windows through WSL (Git Bash has no rsync):
#   wsl -d Ubuntu bash carc/sync.sh push            add -n for a dry run: ... push -n
# The transport is Windows ssh.exe, not WSL's ssh: WSL can't reach discovery through the VPN,
# and ssh.exe already has the `carc` host entry and key from C:\Users\<you>\.ssh\config.
# Never deletes anything on either side. Big data (HingCorpus, Dakshina, results/full) stays home.
set -euo pipefail
cd "$(dirname "$0")/.."

SSH=/mnt/c/Windows/System32/OpenSSH/ssh.exe
[ -x "$SSH" ] || SSH=ssh                       # not under WSL: use the normal ssh
REMOTE=carc:project-repo
OPTS=(-rltvz -e "$SSH" --exclude __pycache__ --exclude logs/)   # -rlt, not -a: Windows file modes would arrive as 777

case "${1:-}" in
  push)
    # sbatch rejects CRLF scripts and bash chokes on them, so convert Windows line endings to LF
    # (in place; LF works fine on Windows too) for every script we're about to send. Data files are left alone.
    find embeddings/*.py embeddings/carc datasets/sentiment/sentimix -type f \
        \( -name "*.sh" -o -name "*.sbatch" -o -name "*.py" \) -not -path "*/logs/*" \
        -exec grep -l $'\r' {} + 2>/dev/null | while read -r f; do sed -i 's/\r$//' "$f"; echo "LF: $f"; done \
        || true                                    # grep exits 1 when nothing has CRLF; that's fine
    rsync "${OPTS[@]}" --relative "${@:2}" \
        embeddings/*.py embeddings/carc datasets/sentiment/sentimix "$REMOTE/"
    ;;
  pull)
    rsync "${OPTS[@]}" "${@:2}" "$REMOTE/embeddings/results/" embeddings/results/
    mkdir -p embeddings/carc/logs
    rsync -rltvz -e "$SSH" "${@:2}" "$REMOTE/embeddings/carc/logs/" embeddings/carc/logs/
    ;;
  *) echo "usage: $0 push|pull [rsync options, e.g. -n for a dry run]"; exit 1 ;;
esac
