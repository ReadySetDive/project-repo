"""How many SentiMix words do hinglishNorm and Dakshina know about?

For every word in SentiMix we ask two questions per resource:
  known        the spelling appears anywhere in the resource
  has variant  the resource also records a *different* spelling of the same word,
               so we could swap it for an attested alternative

"Words" are lowercase alphabetic tokens (no URLs, @mentions, numbers or emoji).
SentiMix tags every token Hin / Eng / other, so results are split by that tag:
Hindi-tagged words are the ones spelling variation is really about.

Two ways of counting:
  types   distinct spellings ("nahi" counts once)
  tokens  occurrences ("nahi" counts every time it's used), i.e. how much of the
          running text is covered

Variants, per resource:
  hinglishNorm  raw -> normalized word pairs from its chat messages, read position by
                position when both sides have the same word count (they do for most
                rows). A word "has a variant" if it was ever paired with a different spelling.
  Dakshina      romanizations of Devanagari words, from the lexicons and the aligned
                romanized Wikipedia sentences. A spelling "has a variant" if its
                Devanagari word was romanized in 2+ different ways.

Usage:  python word_overlap.py          (run from anywhere; reads files next to this script)
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SENTIMIX = HERE.parent / "sentiment" / "sentimix"
HINGLISHNORM = HERE / "hinglishnorm" / "hinglishNorm.json"
DAKSHINA = HERE / "dakshina-hi" / "hi"

WORD = re.compile(r"^[a-z]+$")


def clean(token: str) -> str | None:
    t = token.lower()
    return t if WORD.match(t) else None


# ---------------------------------------------------------------- SentiMix

def sentimix_words(splits):
    """Counter of (word, lang_tag) over the given splits, read straight from the CoNLL files."""
    files = {"train": "train_14k_split_conll.txt", "dev": "dev_3k_split_conll.txt",
             "test": "Hindi_test_unalbelled_conll_updated.txt"}
    counts = Counter()
    for split in splits:
        for line in (SENTIMIX / files[split]).read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2 or parts[0] == "meta":
                continue
            w = clean(parts[0])
            if w:
                lang = parts[1] if parts[1] in ("Hin", "Eng") else "other"
                counts[(w, lang)] += 1
    return counts


# ---------------------------------------------------------------- resources

def load_hinglishnorm():
    vocab, partners = set(), defaultdict(set)
    rows = json.loads(HINGLISHNORM.read_text(encoding="utf-8"))
    aligned = 0
    for r in rows:
        raw = [clean(t) for t in r["inputText"].split()]
        norm = [clean(t) for t in r["normalizedText"].split()]
        vocab.update(w for w in raw + norm if w)
        if len(raw) == len(norm):
            aligned += 1
            for a, b in zip(raw, norm):
                if a and b and a != b:
                    partners[a].add(b)
                    partners[b].add(a)
    print(f"hinglishNorm: {len(rows):,} messages, {aligned / len(rows):.0%} word-aligned; "
          f"{len(vocab):,} distinct words, {len(partners):,} with a recorded variant")
    return vocab, set(partners)


def load_dakshina():
    romans_of = defaultdict(set)                  # Devanagari word -> its romanizations
    for f in sorted((DAKSHINA / "lexicons").glob("*.tsv")):
        for row in csv.reader(open(f, encoding="utf-8"), delimiter="\t"):
            if len(row) >= 2 and clean(row[1]):
                romans_of[row[0]].add(clean(row[1]))
    aligned = DAKSHINA / "romanized" / "hi.romanized.rejoined.aligned.cased_nopunct.tsv"
    for row in csv.reader(open(aligned, encoding="utf-8"), delimiter="\t"):
        if len(row) >= 2 and clean(row[1]):
            romans_of[row[0]].add(clean(row[1]))
    vocab = set().union(*romans_of.values())
    with_variant = set().union(*(r for r in romans_of.values() if len(r) >= 2))
    print(f"Dakshina:     {len(romans_of):,} Devanagari words, {len(vocab):,} distinct romanizations, "
          f"{len(with_variant):,} with a recorded variant")
    return vocab, with_variant


# ---------------------------------------------------------------- report

def pct(part, whole):
    return f"{100 * part / whole:5.1f}%" if whole else "    -"


def coverage_table(title, counts, resources):
    print(f"\n{title}")
    langs = ["Hin", "Eng", "all"]          # SentiMix's other tags (O, EMT) are punctuation, mentions, emoji: never alphabetic words
    header = f"  {'resource':<22} {'measure':<12}" + "".join(f"{l:>16}" for l in langs)
    print(header)
    print(f"  {'':<22} {'':<12}" + "".join(f"{'types / tokens':>16}" for _ in langs))
    print("  " + "-" * (len(header) - 2))

    def rate(vocab, lang, by_tokens):
        items = [(w, n) for (w, l), n in counts.items() if lang in ("all", l)]
        total = sum(n if by_tokens else 1 for _, n in items)
        hit = sum((n if by_tokens else 1) for w, n in items if w in vocab)
        return pct(hit, total)

    for name, (vocab, variant) in resources.items():
        for measure, s in (("known", vocab), ("has variant", variant)):
            cells = "".join(f"{rate(s, l, False):>7} / {rate(s, l, True):>6}" for l in langs)
            print(f"  {name:<22} {measure:<12}{cells}")


def main():
    hn = load_hinglishnorm()
    dk = load_dakshina()
    resources = {"hinglishNorm": hn, "Dakshina": dk, "either (union)": (hn[0] | dk[0], hn[1] | dk[1])}

    for label, splits in (("SentiMix TEST (the set we'd perturb)", ["test"]),
                          ("SentiMix ALL splits", ["train", "dev", "test"])):
        counts = sentimix_words(splits)
        by_lang = Counter()
        for (_, l), n in counts.items():
            by_lang[l] += n
        print(f"\n=== {label}: {len({w for w, _ in counts}):,} distinct words, {sum(counts.values()):,} word tokens "
              f"(Hin {by_lang['Hin']:,} / Eng {by_lang['Eng']:,})")
        coverage_table("Coverage (types = distinct spellings, tokens = share of running text)", counts, resources)

    # what are we missing? the most frequent Hindi-tagged test words with no variant anywhere
    counts = sentimix_words(["test"])
    union_var = hn[1] | dk[1]
    missing = Counter()
    for (w, l), n in counts.items():
        if l == "Hin" and w not in union_var:
            missing[w] += n
    print("\nMost frequent Hindi-tagged TEST words with no recorded variant in either resource:")
    print("  " + ", ".join(f"{w} ({n})" for w, n in missing.most_common(30)))


if __name__ == "__main__":
    main()
