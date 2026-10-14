# hinglishNorm

- Source: https://github.com/piyushmakhija5/hinglishNorm (`dataset/`), fetched 2026-10-10
- Paper: Makhija, Kumar & Gupta. 2020. *hinglishNorm - A Corpus of Hindi-English Code Mixed Sentences for Text Normalization.* COLING 2020 (Industry Track), pp. 136–145. https://aclanthology.org/2020.coling-industry.13
- License: CC BY-NC-SA 4.0 (stated in the upstream README; the repo's `license/` folder only holds badge images)
- Files: `hinglishNorm.json` (all 13,494 rows), `hinglishNorm_trainSet.json`, `hinglishNorm_testSet.json`, upstream `README.md`
- Row format: `{id, inputText, tags, normalizedText}`; `tags` is a stringified Python list, one tag per input token (off by one in ~15% of rows)
