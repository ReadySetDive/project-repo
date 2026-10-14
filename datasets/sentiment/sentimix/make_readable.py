"""Unwind the SentiMix CoNLL files into readable CSVs: id, sentiment, phrase.

The CoNLL files store one token per line, and the tokenizer split URLs, mentions,
hashtags and contractions apart ("https // t . co / X", "@ Nehr _ who", "aren ’ t").
This rebuilds each tweet as readable text. It's for reading and spreadsheets;
model code should parse the CoNLL files directly, since the rebuild is approximate.

Usage:  python make_readable.py         (writes readable/train.csv, dev.csv, test.csv)
"""

import csv
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "readable"
SPLITS = {
    "train": "train_14k_split_conll.txt",
    "dev": "dev_3k_split_conll.txt",
    "test": "Hindi_test_unalbelled_conll_updated.txt",
}


MOJIBAKE = re.compile("[ÃÂâàðÙØ]")


def unmojibake(tok: str) -> str:
    """Undo UTF-8 text that was decoded as cp1252 ("â€¦" -> "…"). Only the test file has this.
    Returns the token unchanged if it doesn't round-trip cleanly."""
    if not MOJIBAKE.search(tok):
        return tok
    raw = bytearray()
    for ch in tok:
        try:
            raw += ch.encode("cp1252")
        except UnicodeEncodeError:
            if ord(ch) < 256:      # bytes cp1252 leaves undefined (0x81, 0x8d, 0x90, ...) passed through as-is
                raw.append(ord(ch))
            else:
                return tok
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return tok


def parse(path: Path):
    """Yield (id, label or None, [tokens]) per tweet. Headers are lines whose first field is exactly 'meta'."""
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if parts[0] == "meta" and len(parts) >= 2:
            if cur:
                yield cur
            cur = (parts[1], parts[2] if len(parts) > 2 else None, [])
        elif cur and parts[0].strip():
            cur[2].append(unmojibake(parts[0]))
    if cur:
        yield cur


def detok(tokens: list[str]) -> str:
    s = " ".join(tokens)
    # URLs: "https // t . co / X", "https // tco / X", "https // t co / X"
    s = re.sub(r"(https?)\s*//\s*t\s*\.?\s*co\s*/\s*(\S+)", r"\1://t.co/\2", s)
    s = re.sub(r"(https?)\s*//\s*", r"\1://", s)
    # mentions and hashtags, including underscores split into separate tokens
    s = re.sub(r"([@#])\s+(\w+(?:\s+_\s+\w+|\s+_(?=\s|$))*)",
               lambda m: m.group(1) + re.sub(r"\s+", "", m.group(2)), s)
    # contractions: aren ’ t, it ’ s, I ' m
    s = re.sub(r"(\w)\s+([’'])\s+(t|s|re|ve|ll|d|m)\b", r"\1\2\3", s)
    # no space before closing punctuation, none after opening brackets
    s = re.sub(r"\s+([.,!?;:…)\]%])", r"\1", s)
    s = re.sub(r"([(\[])\s+", r"\1", s)
    return s.strip()


def main():
    OUT.mkdir(exist_ok=True)
    test_labels = {}
    with (HERE / "test_labels_hinglish.txt").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_labels[row["Uid"].strip()] = row["Sentiment"].strip()

    for split, fname in SPLITS.items():
        rows = []
        for tid, label, toks in parse(HERE / fname):
            if label is None:
                label = test_labels.get(tid, "")
            rows.append((tid, label, detok(toks)))
        missing = sum(1 for _, lab, _ in rows if not lab)
        # utf-8-sig so Excel shows emoji and Devanagari correctly
        with (OUT / f"{split}.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "sentiment", "phrase"])
            w.writerows(rows)
        print(f"{split:5}  {len(rows):>6} tweets  -> readable/{split}.csv"
              + (f"  ({missing} missing labels!)" if missing else ""))


if __name__ == "__main__":
    main()
