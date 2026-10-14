"""How many SentiMix tweets appear in a pretraining corpus (e.g. L3Cube-HingCorpus)?

HingBERT was pretrained on HingCorpus (52.9M Hinglish sentences scraped from Twitter);
SentiMix is Hinglish Twitter from about the same time. If SentiMix test tweets are in the
corpus, part of HingBERT's score could come from having read them.

Two checks per SentiMix tweet, after normalizing both sides (lowercase, drop URLs,
@mentions, "rt", punctuation and emoji, collapse whitespace):
  exact     the whole tweet equals a corpus line
  coverage  share of the tweet's word 5-grams found anywhere in the corpus; this catches
            tweets the corpus split into sentences or lightly edited. Only for tweets with
            at least 5 words; shorter ones get the exact check only.

Calibrated on a fake corpus of planted SentiMix tweets (100 verbatim, 100 split in half,
5k noise lines): n=5, threshold 0.5 caught 100/100 verbatim and 89/100 split, 3/2800
false positives (n=8 caught only 23/100 split). A 52M-line corpus has many more common
phrases, so coverage hits are noisier than exact ones: read the printed samples.

Reads plain text, .gz, .tar.gz/.tgz or .zip (every member), one sentence per line. Stdlib only.

Usage:  python check_contamination.py <corpus file or dir> [--n 5] [--threshold 0.5]
Writes contamination.csv (id, split, words, exact, coverage) next to this script.
"""

import argparse
import csv
import gzip
import io
import re
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import make_readable as readable

HERE = Path(__file__).resolve().parent
URL = re.compile(r"https?\s*:?\s*//\S*|\bwww\.\S+|\bt\.co/\S*")
MENTION = re.compile(r"@\s*\w+")
NONWORD = re.compile(r"[^\w\s]|_")


def normalize(s: str) -> list[str]:
    s = URL.sub(" ", s.lower())
    s = MENTION.sub(" ", s)
    s = NONWORD.sub(" ", s)
    words = s.split()
    if words and words[0] == "rt":
        words = words[1:]
    return words


def corpus_lines(path: Path):
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        if f.name in (".gitignore", "SOURCE.md"):
            continue
        if f.name.endswith((".tar.gz", ".tgz", ".tar")):
            with tarfile.open(f, "r:*") as tar:           # streamed; members are never extracted to disk
                for m in tar:
                    if not m.isfile():
                        continue
                    print(f"  reading {f.name}:{m.name} ({m.size / 1e9:.1f} GB)", file=sys.stderr)
                    raw = tar.extractfile(m)
                    if raw:
                        yield from io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        elif f.suffix == ".zip":
            with zipfile.ZipFile(f) as z:
                for m in z.namelist():
                    if m.endswith("/"):
                        continue
                    print(f"  reading {f.name}:{m}", file=sys.stderr)
                    with z.open(m) as raw:
                        yield from io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        else:
            print(f"  reading {f.name}", file=sys.stderr)
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt", encoding="utf-8", errors="replace") as fh:
                yield from fh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--n", type=int, default=5, help="n-gram size in words")
    ap.add_argument("--threshold", type=float, default=0.5, help="coverage counted as contaminated")
    ap.add_argument("--max-lines", type=int, help="stop early (testing)")
    args = ap.parse_args()

    # SentiMix tweets, normalized
    test_ids = {line.split(",")[0].strip()
                for line in (HERE / "test_labels_hinglish.txt").read_text(encoding="utf-8").splitlines()[1:]}
    tweets = []                                   # (id, split, words)
    for split, fname in readable.SPLITS.items():
        for tid, _, toks in readable.parse(HERE / fname):
            tweets.append((tid, split, normalize(readable.detok(toks))))
    assert test_ids <= {t[0] for t in tweets if t[1] == "test"}

    exact_index = defaultdict(list)               # normalized tweet -> tweet indices
    gram_index = defaultdict(list)                # n-gram -> [(tweet index, gram position)]
    n_grams = []
    for i, (_, _, w) in enumerate(tweets):
        if w:
            exact_index[" ".join(w)].append(i)
        grams = [tuple(w[j:j + args.n]) for j in range(len(w) - args.n + 1)]
        n_grams.append(len(grams))
        for j, g in enumerate(grams):
            gram_index[g].append((i, j))
    hit_grams = [set() for _ in tweets]
    exact = [False] * len(tweets)
    print(f"{len(tweets)} SentiMix tweets, {sum(1 for c in n_grams if c)} with >= {args.n} words, "
          f"{len(gram_index)} distinct {args.n}-grams", file=sys.stderr)

    t0, lines = time.time(), 0
    for line in corpus_lines(args.corpus):
        lines += 1
        w = normalize(line)
        if not w:
            continue
        for i in exact_index.get(" ".join(w), ()):
            exact[i] = True
        for j in range(len(w) - args.n + 1):
            hits = gram_index.get(tuple(w[j:j + args.n]))
            if hits:
                for i, pos in hits:
                    hit_grams[i].add(pos)
        if lines % 2_000_000 == 0:
            print(f"  {lines / 1e6:.0f}M lines, {time.time() - t0:.0f}s", file=sys.stderr)
        if args.max_lines and lines >= args.max_lines:
            break
    print(f"scanned {lines:,} corpus lines in {(time.time() - t0) / 60:.1f} min\n", file=sys.stderr)

    rows = []
    for i, (tid, split, w) in enumerate(tweets):
        cov = len(hit_grams[i]) / n_grams[i] if n_grams[i] else None
        rows.append((tid, split, len(w), exact[i], cov))
    with (HERE / "contamination.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "split", "words", "exact", "coverage"])
        wr.writerows((r[0], r[1], r[2], int(r[3]), "" if r[4] is None else f"{r[4]:.3f}") for r in rows)

    print(f"{'split':6} {'tweets':>7} {'exact':>14} {'cov>=' + str(args.threshold):>14} {'either':>14} {'cov>0':>14}")
    for split in readable.SPLITS:
        rs = [r for r in rows if r[1] == split]
        n = len(rs)
        ex = sum(r[3] for r in rs)
        cv = sum(1 for r in rs if r[4] is not None and r[4] >= args.threshold)
        ei = sum(1 for r in rs if r[3] or (r[4] is not None and r[4] >= args.threshold))
        any_ = sum(1 for r in rs if r[4])
        pct = lambda k: f"{k:>6} ({100 * k / n:4.1f}%)"
        print(f"{split:6} {n:>7} {pct(ex):>14} {pct(cv):>14} {pct(ei):>14} {pct(any_):>14}")
    # eyeball check: coverage-only hits (not exact) from test, spread across the coverage range
    text = {(tid, split): " ".join(w) for tid, split, w in tweets}
    cov_only = sorted((r for r in rows if r[1] == "test" and not r[3] and r[4] is not None
                       and r[4] >= args.threshold), key=lambda r: -r[4])
    if cov_only:
        print("\nsample coverage-only test hits (real copies or common phrases?):")
        for r in cov_only[:: max(1, len(cov_only) // 10)][:10]:
            print(f"  {r[4]:.2f}  {text[(r[0], r[1])][:110]}")
    print(f"\nper-tweet results: {HERE / 'contamination.csv'}")


if __name__ == "__main__":
    main()
