# SentiMix (Hinglish)

- **Source:** https://huggingface.co/datasets/RTT1/SentiMix, a re-upload of the original SemEval-2020 Task 9 Hindi-English files. Fetched 2026-10-10.
- **Paper:** Patwa et al. 2020. *SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets.* https://aclanthology.org/2020.semeval-1.100. Cite this, not the HF re-upload.
- **License:** the HF card says "openrail", but the provenance is the SemEval release, so treat it as research use and cite the task paper.
- **Annotation:** each tweet was labeled by 2 annotators out of a pool of ~60. Raw agreement was 55%, and disagreements were discarded, so every released label is one both annotators agreed on (paper §4.1).

| File | Tweets | Labels |
|---|---|---|
| `train_14k_split_conll.txt` | 14,000 | in file |
| `dev_3k_split_conll.txt` | 3,000 | in file |
| `Hindi_test_unalbelled_conll_updated.txt` | 3,000 | in `test_labels_hinglish.txt` (`Uid,Sentiment`) |

**Format:** CoNLL-style. Each tweet starts with a header line `meta<TAB><id><TAB><label>`; the test header has no label. Then comes one `token<TAB><lang>` line per token, where lang is `Hin`, `Eng`, `O` or `EMT`. Tweets are separated by blank lines.

**Gotchas:**
- Split headers on the exact first field `meta`. A few real tokens also start with "meta" (`metal`, `metabolism`), so a naive `grep ^meta` over-counts.
- Text is pre-tokenized: URLs appear as `https // t . co / …` and mentions as `@ handle`. Rebuild the tweet text before using it with transformers.
- **The test file is mojibake.** In `Hindi_test_unalbelled_conll_updated.txt`, UTF-8 was decoded as cp1252, so `…` became `â€¦`, emoji became `ðŸ...`, and Gurmukhi/Devanagari came out garbled. 2,661 lines are affected; train and dev are clean. Repair the test tokens before evaluating, or models will see garbled emoji and punctuation only at test time. `make_readable.py` has the `unmojibake()` fix, and its output has 0 garbled rows left.
- Some rows aren't Hinglish (junk and other languages).
- Class balance (train): neutral 5,264, positive 4,634, negative 4,102.
- IDs don't overlap across splits, and 1 test tweet text also appears in train.
