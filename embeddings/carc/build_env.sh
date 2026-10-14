#!/bin/bash
# Build the training env with plain venv + pip (no uv), from requirements-lock.txt.
# Everything lives on /home1: uv's cache on /scratch1 wedged on Oct 10 (reads hang in D state).
#
#   bash embeddings/carc/build_env.sh fetch    download every wheel once into $WHEELS (needs internet)
#   bash embeddings/carc/build_env.sh install  build $ENV from $WHEELS, offline, then smoke-test it
#   bash embeddings/carc/build_env.sh bundle   tar interpreter + wheels + lock + this script into $BUNDLE
#   bash embeddings/carc/build_env.sh all      fetch, install
#
# The bundle rebuilds the identical env on any x86-64 Linux box, offline:
#   tar xf hing-env-bundle.tar && PYTHON=$PWD/python/bin/python3.12 WHEELS=$PWD/wheels \
#     bash build_env.sh install
# Run from the repo root on a login node (the transfer node had a stuck rm on Oct 11).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LOCK=${LOCK:-$HERE/requirements-lock.txt}
ENV=${ENV:-$HOME/venvs/hing2}
WHEELS=${WHEELS:-$HOME/wheelhouse/hing}
BUNDLE=${BUNDLE:-$HOME/hing-env-bundle.tar}
# Interpreter: a standalone CPython (relocatable), by default the one uv downloaded to /home1.
# Only the python binary is used, never uv itself.
PYTHON=${PYTHON:-$(ls -d "$HOME"/.local/share/uv/python/cpython-3.12.*-linux-x86_64-gnu/bin/python3.12 2>/dev/null | tail -1)}
[ -x "$PYTHON" ] || { echo "no python 3.12 found; set PYTHON=/path/to/python3.12"; exit 1; }
PIP_OPTS=(--no-cache-dir --timeout 120 --retries 10 --disable-pip-version-check)

fetch() {
    mkdir -p "$WHEELS"
    "$PYTHON" -m venv --clear "$HOME/.cache/hing-pip-tool"            # throwaway env just to run pip
    "$HOME/.cache/hing-pip-tool/bin/pip" download "${PIP_OPTS[@]}" --no-deps -d "$WHEELS" -r "$LOCK" \
        --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu128
    echo "wheels: $(ls "$WHEELS" | wc -l) files, $(du -sh "$WHEELS" | cut -f1) in $WHEELS"
}

install() {
    "$PYTHON" -m venv --clear "$ENV"
    "$ENV/bin/pip" install "${PIP_OPTS[@]}" --no-index --no-deps --find-links "$WHEELS" -r "$LOCK"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$ENV/bin/python" -c "import torch, transformers, accelerate, datasets, sklearn; \
print('ok', torch.__version__, transformers.__version__, 'cuda build', torch.version.cuda)"
    echo "env ready: $ENV/bin/python"
}

bundle() {
    local pyroot; pyroot=$(dirname "$(dirname "$PYTHON")")
    local stage; stage=$(mktemp -d)
    ln -s "$pyroot" "$stage/python"; ln -s "$WHEELS" "$stage/wheels"
    cp "$LOCK" "$0" "$stage/"
    # record of the env the bundle came from (should match the lock exactly)
    { echo "# pip freeze of $ENV on $(hostname), $(date)"; "$ENV/bin/pip" freeze; } > "$stage/frozen.txt" 2>/dev/null || true
    tar -chf "$BUNDLE" -C "$stage" python wheels frozen.txt "$(basename "$LOCK")" "$(basename "$0")"   # -h: follow the links
    rm -rf "$stage"
    echo "bundle: $BUNDLE ($(du -sh "$BUNDLE" | cut -f1))"
}

case "${1:-}" in
    fetch) fetch ;;
    install) install ;;
    bundle) bundle ;;
    all) fetch; install ;;
    *) echo "usage: $0 fetch|install|bundle|all"; exit 1 ;;
esac
