# Perturbation sources

Real, human-written Hinglish spelling variants. We use them to respell words in the SentiMix test set instead of inventing rules. Neither has sentiment labels.

| | hinglishNorm (`hinglishnorm/`) | Dakshina Hindi (`dashnika-hi/`) |
|---|---|---|
| What | 13,494 chat messages, each raw → human-normalized, one tag per token | 30k Devanagari words, each romanized by 3+ native speakers (with counts), plus 10k romanized Wikipedia sentences |
| Spellings | Organic and messy, with texting forms (`nhi`, `kr`, `plz`, `bohot`) | Elicited and clean (`nahin`/`nahi`, `accha`/`achcha`), no texting forms |
| License | CC BY-NC-SA 4.0 | CC BY-SA 4.0 |
| Watch out | Normalization is inconsistent, and short ambiguous tokens (`ur`, `h`, `ni`) glue clusters together (see `hinglishnorm/graph/`) | Too few annotators per word to capture common shortcuts |

**Compared to the other data**
- **SentiMix** (SemEval-2020 Task 9): the labeled data we train and test on (14k/3k/3k tweets, 3-class). These two only supply the respellings. Dakshina covers about 73% of SentiMix's Hindi tokens. hinglishNorm adds the texting forms Dakshina misses.
- **Hinglish YouTube comments** (HF `shae2977/hinglish-youtube-sentiments-dataset`, 3,190, CC BY 4.0) and **IIITH Facebook** (Joshi et al. 2016, 3,879, no license): labeled sets with naturally different spelling habits. Use them for test-only cross-dataset evaluation, not as perturbation sources. Not downloaded yet.
- **Skipped:** Aksharantar (mostly machine-mined), the UD Hindi-English treebank (tweet text stripped), and several HF "Hinglish sentiment" sets that are LLM-generated or repackaged SentiMix.
