# L3Cube-HingCorpus (roman)

- **Source:** https://github.com/l3cube-pune/code-mixed-nlp. The Google Drive link is in that README (file id `1s_6eHO9zDhxQ-xVN1TyNguszV1meZkl9`). Fetched 2026-10-10 with `uvx gdown <id>`.
- **Paper:** Nayak & Joshi 2022, *L3Cube-HingCorpus and HingBERT: A Code Mixed Hindi-English Dataset and BERT Language Models.* This is the corpus HingBERT was pretrained on.
- **License:** CC BY-NC-SA 4.0. Non-commercial only. Anything released that is derived from it (e.g. pretrained weights) must carry the same license.
- **Size:** 52.93M sentences, 1.04B tokens, Hinglish Twitter in roman script. There are no tweet IDs or dates.

| File | Size |
|---|---|
| `L3Cube-HingCorpus_roman.tar.gz` | 2.5 GB |
| ↳ `R11_final_data/concatenated_train_final_shuffled.txt` | 4.9 GB |
| ↳ `R11_final_data/concatenated_validation.txt` | 0.5 GB |

**Format:** one tweet per line, with @mentions already removed. It contains junk lines (symbols only) and emoji.

The archive is gitignored. Scripts read the `.tar.gz` directly (streamed, never extracted). See NOTES.md for the SentiMix contamination check.
