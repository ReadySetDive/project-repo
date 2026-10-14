# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Paired significance test between the models run by run_mteb.py.

Reads the per-item predictions from results/<protocol>/predictions/ and the gold
labels run_mteb.py saved to results/<protocol>/gold/ (for older SentiMix runs with no
gold file, it rebuilds them from the CoNLL files), and for every pair of
models runs a paired bootstrap on macro-F1 (resampling tweets, both models scored
on the same resample) plus McNemar's test on per-tweet correctness.

Usage:  uv run embeddings/compare.py [--task sentimix|urdu-roman] [--protocol full] [--split test]
"""

import argparse
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SENTIMIX = HERE.parent / "datasets" / "sentiment" / "sentimix"
LABELS = ["negative", "neutral", "positive"]   # must match run_mteb.py
TASKS = {"sentimix": "SentiMixHinglishClassification",
         "urdu-roman": "UrduRomanSentimentClassification.v2"}

_spec = importlib.util.spec_from_file_location("make_readable", SENTIMIX / "make_readable.py")
assert _spec and _spec.loader
readable = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readable)


def gold(out: Path, task: str, split: str) -> list:
    saved = out / "gold" / f"{task}_{split}.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    if task != TASKS["sentimix"]:
        raise SystemExit(f"no {saved}; rerun run_mteb.py for this task")
    test_labels = {}
    for line in (SENTIMIX / "test_labels_hinglish.txt").read_text(encoding="utf-8").splitlines()[1:]:
        uid, lab = line.split(",")
        test_labels[uid.strip()] = lab.strip()
    return [LABELS.index(lab or test_labels[tid])
            for tid, lab, _ in readable.parse(SENTIMIX / readable.SPLITS[split])]


def macro_f1(y: np.ndarray, p: np.ndarray, idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Macro-F1 for each bootstrap row of idx (shape boot x n), vectorized."""
    yb, pb = y[idx], p[idx]
    f1s = []
    for c in range(n_classes):
        tp = ((yb == c) & (pb == c)).sum(1)
        fp = ((yb != c) & (pb == c)).sum(1)
        fn = ((yb == c) & (pb != c)).sum(1)
        f1s.append(np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
    return np.mean(f1s, axis=0)


def mcnemar(y, a, b) -> tuple[int, int, float]:
    """Exact-ish McNemar (chi-square with continuity correction). Returns (a-only right, b-only right, p)."""
    ra, rb = a == y, b == y
    n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    if n01 + n10 == 0:
        return n01, n10, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return n01, n10, math.erfc(math.sqrt(chi2 / 2))   # chi2 with 1 df -> p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="sentimix", choices=list(TASKS))
    ap.add_argument("--protocol", default="full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=544)
    args = ap.parse_args()

    task = TASKS[args.task]
    out = HERE / "results" / args.protocol
    gold_raw = gold(out, task, args.split)
    raw = {}
    for f in sorted((out / "predictions").glob(f"*/{task}_predictions.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        runs = d["default"][args.split]
        if len(runs) != 1:
            print(f"note: {f.parent.name} has {len(runs)} experiments; using the first")
        if len(runs[0]) != len(gold_raw):
            print(f"skip {f.parent.name}: {len(runs[0])} predictions vs {len(gold_raw)} gold (a --limit run?)")
            continue
        raw[d["mteb_model_meta"]["model_name"]] = runs[0]
    if not raw:
        raise SystemExit(f"no {task} predictions found under {out / 'predictions'}")
    # labels may be ints or strings depending on the task; map both sides to 0..k-1
    classes = sorted({str(v) for v in gold_raw} | {str(v) for r in raw.values() for v in r})
    code = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    y = np.array([code[str(v)] for v in gold_raw])
    preds = {name: np.array([code[str(v)] for v in r]) for name, r in raw.items()}

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(y), size=(args.boot, len(y)))
    full = np.arange(len(y))[None, :]

    print(f"{task} {args.split}: n={len(y)}, {args.boot} paired bootstrap resamples\n")
    print(f"{'model':32} {'macroF1':>8} {'95% CI':>17} {'acc':>7}")
    boots = {}
    for name, p in preds.items():
        boots[name] = macro_f1(y, p, idx, k)
        lo, hi = np.percentile(boots[name], [2.5, 97.5])
        print(f"{name:32} {macro_f1(y, p, full, k)[0]:8.4f}   [{lo:.4f}, {hi:.4f}] {(p == y).mean():7.4f}")

    print(f"\n{'A vs B':62} {'dF1 (A-B)':>9} {'95% CI':>19} {'p_boot':>7} {'McNemar A/B-only, p':>24}")
    for a, b in itertools.combinations(preds, 2):
        d = boots[a] - boots[b]
        obs = macro_f1(y, preds[a], full, k)[0] - macro_f1(y, preds[b], full, k)[0]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        n01, n10, p_mc = mcnemar(y, preds[a], preds[b])
        print(f"{a + ' vs ' + b:62} {obs:+9.4f}   [{lo:+.4f}, {hi:+.4f}] {p_boot:7.4f} {n01:>8}/{n10:<6} {p_mc:.2g}")


if __name__ == "__main__":
    main()
