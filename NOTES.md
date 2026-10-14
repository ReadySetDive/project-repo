# Experiment notes

Running log of results and decisions. Newest section last. All F1 scores are macro-F1 on the SentiMix (SemEval-2020 Task 9) Hinglish test set (3,000 tweets) unless noted.

## 2026-10-10: frozen embeddings (MTEB linear probe)

`embeddings/run_mteb.py --protocol full`: frozen model, logistic regression trained on all 14k train tweets. Mean pooling for HingBERT and mmBERT (not in the MTEB registry); F2LLM uses its registry instruction prompt and last-token pooling.

| Model | SentiMix | Roman Urdu (MTEB `UrduRomanSentimentClassification.v2`, n=957) |
|---|---|---|
| l3cube-pune/hing-bert (2022, 110M) | **0.709** | **0.586** |
| jhu-clsp/mmBERT-base (2025, 307M) | 0.656 | 0.505 |
| codefuse-ai/F2LLM-v2-0.6B (2026, 600M) | 0.634 | 0.523 |

- SentiMix: every pair is significant (paired bootstrap + McNemar, `embeddings/compare.py`). HingBERT beats mmBERT by +5.3 and F2LLM by +7.5 (p < 1e-7). mmBERT beats F2LLM by +2.2 (p = 0.03).
- Roman Urdu: HingBERT beats mmBERT by +8.1 and F2LLM by +6.3 (p ≤ 0.002). F2LLM vs mmBERT is not significant.
- MTEB(Indic, v1) classification is native script except one task (Roman Urdu), and none of it is code-mixed. F2LLM ranking top for sub-1B Indic says nothing about romanized code-mixed text.
- Frozen probing understates models that weren't trained as sentence embedders, and F2LLM's retrieval embedding encodes topic more than sentiment. Use fine-tuning for model comparisons.

## 2026-10-10: full fine-tuning

`embeddings/finetune.py` on CARC (1× A40, debug partition, jobs 12917727 and 12918372). Every weight is updated plus a new classification head. 3 epochs, batch 32, max length 128, 10% warmup, linear decay, bf16 autocast over fp32 weights. The best epoch is picked on dev and test is scored once. Seed 544, **single seed**. Learning rates are per-family defaults, not tuned.

| Model | Type | LR | Dev | **Test** | Acc | Train min | Frozen → FT |
|---|---|---|---|---|---|---|---|
| HingBERT | encoder + Hinglish pretraining | 2e-5 | 0.664 | **0.730** | 0.728 | 1.1 | +2.1 |
| F2LLM-v2-0.6B | decoder + embedding training | 1e-5 | 0.635 | **0.713** | 0.710 | 8.8 | +7.9 |
| mmBERT-base | encoder, multilingual | 5e-5 | 0.639 | **0.692** | 0.688 | 2.2 | +3.6 |
| Qwen3-0.6B-Base | plain decoder (F2LLM's base) | 1e-5 | 0.624 | **0.689** | 0.685 | 8.3 | n/a |

Significance (test-set bootstrap, n=3000; this does **not** include seed variance):
- HingBERT − F2LLM +1.7 [+0.2, +3.2], p = 0.02
- F2LLM − mmBERT +2.1 [+0.4, +3.8], p = 0.02
- HingBERT − mmBERT +3.8 [+2.2, +5.4], p < 1e-5
- Qwen3-Base has not been added to `compare.py` yet (its predictions are still on CARC).

Readings:
- Fine-tuning moved F2LLM from last to second; the frozen probe was misleading for it.
- F2LLM's embedding stage is worth about +2.4 over plain Qwen3-Base under fine-tuning (same architecture and size).
- 600M plain decoder (Qwen3-Base) ≈ 307M encoder (mmBERT). That fits the controlled finding that encoders beat decoders at classification at equal size and data (Ettin suite, Weller et al. 2025).
- The 110M HingBERT beats a 5.5× larger decoder: in-domain pretraining plus an encoder architecture.
- **Dev scores 5–8 points below test for all four models.** It's a property of the SentiMix splits, not the models. Mention it in the write-up.
- Gaps of 1.7–2.1 are the size of typical seed variance (±0.5–1.5). **Run multiple seeds before claiming a ranking.** HingBERT vs mmBERT (+3.8) is safe.

Saved models (best epoch, weights + tokenizer) on CARC scratch, which is not backed up and gets purged:
`/scratch1/mgordy/sentimix-models/12917727/{l3cube-pune__hing-bert, jhu-clsp__mmBERT-base, codefuse-ai__F2LLM-v2-0.6B}`, `/scratch1/mgordy/sentimix-models/12918372/Qwen__Qwen3-0.6B-Base`.
Keep these. The respelling robustness test has to use the same models that produced the clean scores, because retraining isn't bit-for-bit reproducible.

## 2026-10-10: HingCorpus contamination check

Question: was HingBERT pretrained on SentiMix test tweets? Both come from Hinglish Twitter at about the same time.

`datasets/sentiment/sentimix/check_contamination.py` scanned all 52,933,356 lines of L3Cube-HingCorpus (roman) in 16.5 minutes on a laptop CPU. Both sides are normalized (lowercase; URLs, @mentions, punctuation and emoji dropped). Two checks: exact whole-tweet match, and the share of the tweet's word 5-grams found anywhere in the corpus. The check was calibrated on planted copies: n=5 with threshold 0.5 caught 100% of verbatim copies and 89% of tweets split in half, with about 0.1% false positives.

| Split | Exact | Exact or 5-gram coverage ≥ 0.5 | Any 5-gram shared |
|---|---|---|---|
| train (14k) | 35 (0.2%) | 227 (1.6%) | 31.6% |
| dev (3k) | 9 (0.3%) | 62 (2.1%) | 31.5% |
| test (3k) | 9 (0.3%) | 62 (2.1%) | 32.1% |

The flagged tweets are mostly stock phrases, song lyrics and proverbs ("all the best team india jeet ke aana", "kutte ki dhum kabhi seedhi nahi hoti"), not copied tweets. The ~32% that share any 5-gram is ordinary Hinglish phrasing.

Fine-tuned F1 without the 62 flagged test tweets:

| Model | All 3,000 | Clean 2,938 | Flagged 62 |
|---|---|---|---|
| HingBERT | 0.7303 | 0.7295 | 0.701 |
| F2LLM | 0.7131 | 0.7125 | 0.663 |
| mmBERT | 0.6922 | 0.6909 | 0.668 |

**Conclusion:** contamination is negligible. Removing the flagged tweets changes no score by more than 0.0015, and HingBERT does worse on the flagged tweets than overall, so its lead isn't memorization.

Sentence for the paper: *Only 0.3% of SentiMix test tweets occur verbatim in HingCorpus (2.1% including partial 5-gram overlap); excluding them changes no model's macro-F1 by more than 0.002.*

Per-tweet flags: `datasets/sentiment/sentimix/contamination.csv` (id, split, words, exact, coverage).

**Decision (2026-10-10): handle contamination at evaluation, not in the corpus.** HingmmBERT pretraining (`embeddings/pretrain_mlm.py`) uses the **unfiltered** corpus, exactly as HingBERT did, so the two models get the same data. **TODO, filter later:** for every model (and especially HingBERT vs HingmmBERT), report macro-F1 on the full 3,000 test tweets *and* on the 2,938-tweet clean subset (drop test rows flagged exact or coverage ≥ 0.5 in `contamination.csv`), the same way as the table above. If a reviewer wants a corpus-side filter instead, rebuild it from `check_contamination.py`'s `normalize()`: drop corpus lines that exactly match a flagged dev/test tweet or share 2+ of its word 5-grams. A version of this was written and tested, then removed. Nothing was committed, so it isn't in git.

## Decisions and open questions

- **Add a modern model to the lineup.** F2LLM-v2-0.6B is the modern off-the-shelf model; consider swapping XLM-R for mmBERT.
- **HingBERT recipe** (Nayak & Joshi 2022): 2 epochs over HingCorpus (≈2B tokens of exposure), lr 1e-5, batch 64. HingmmBERT pilot plan: 500M tokens (25% of that exposure), checkpoints at 100M/200M/500M, lr 5e-5 (stated deviation, for a smaller budget), batch 64, unfiltered corpus. **Go/no-go rule, set before seeing results:** single-seed fine-tune ≥ ~0.75 means strong, take it to the group; 0.73–0.75 needs seeds; below HingBERT (0.730) is a different story, so check the checkpoint trend. Speed benchmark (`embeddings/carc/pretrain_benchmark.sbatch`) comes first, to replace the ~30k tokens/s A40 guess with a measurement.
- **Domain-pretraining ablation:** prefer **HingmmBERT** (continued MLM pretraining of mmBERT on HingCorpus) over HingQwen. It isolates Hinglish exposure on a modern encoder, avoids the decoder handicap, and costs about half as much. Plan: a 100–200M-token subset first. Estimates are extrapolated from measured throughput: about 1.5–3k SU for a 0.6B model on an A100, less for mmBERT. Clear SU use with the group first.
- **Seed and learning-rate sweep** (equal budget per model): 4 learning rates, max 5 epochs, best on dev, then 5 seeds of the winner; report mean ± std. About 2.5–3k SU on the gpu partition.
- **LP-FT** (linear probe, then fine-tune; Kumar et al. 2022) is a candidate extra condition. It's claimed to preserve out-of-distribution robustness, which is relevant to the respelling test.
- **Core experiment still to build:** the attested-respelling test set (hinglishNorm and Dakshina variants), scored with the saved models.

## Compute log (CARC, account xiangren_1987)

| Job | What | GPU | Train time | Est. SU |
|---|---|---|---|---|
| 12917714 | failed at once (submitted from the wrong directory) | A40 debug | n/a | ~0 |
| 12917727 | HingBERT + mmBERT + F2LLM fine-tunes | A40 debug | ~12 min | ~250 (check with `jobinfo`) |
| 12918372 | Qwen3-0.6B-Base fine-tune | A40 debug | 8.3 min | ~180 (check with `jobinfo`) |
| 12919395 | HingmmBERT pretraining benchmark: OOM in ~3 min (sparse MLM head flag set after construction; fixed) | A40 debug | n/a | ~60 |
| 12919649 | HingmmBERT pretraining benchmark, ended ~9 min in | A40 debug | ~8 min | ~180 |
| 12920330 | HingmmBERT 200M pilot: FAILED after 6.5 min on a V100 ("no kernel image": torch 2.11+cu128 supports sm_75+ only, V100 is sm_70) | V100 gpu | n/a | ~100 |

| 12920724 | HingmmBERT 200M pilot, cancelled while pending (replaced by 600M) | n/a | n/a | 0 |
| 12920820 | HingmmBERT 600M pilot, cancelled while pending (replaced by 1B) | n/a | n/a | 0 |
| 12920851 | **HingmmBERT 1B pilot**: `--time 12:00:00`, `EXTRA="--tokens 1B --save-at 100M,200M,400M,600M,800M,1B"`, lr 5e-5 linear decay over 1B; submitted 2026-10-10 ~18:25. ~50% of HingBERT's ~2B-token exposure | a40/l40s/a100 | pending | est. ~12.6k on A40 (10.5 h) |

| 12920851 → **FAILED in 2 s** at 20:29 on an L40S (it got a GPU early!): `uv: command not found`. I had submitted it with `ssh carc "sbatch ..."`, a non-interactive shell without `~/.local/bin` on PATH, and `--export=ALL` carried that PATH into the job. Also, `--export=EXTRA='... 100M,200M,...'` would have cut the checkpoint list at the first comma (`--export` splits on commas). Fixes: every sbatch now does `export PATH="$HOME/.local/bin:$PATH"`, and the pilot takes overrides as trailing args (`"$@"`) | L40S | 2 s | ~1 |
| 12922539 | **HingmmBERT 1B pilot, resubmitted** 20:35: `sbatch --time=12:00:00 embeddings/carc/pretrain_pilot.sbatch --tokens 1B --save-at 100M,200M,400M,600M,800M,1B` | a40/l40s/a100 | pending | est. ~12.6k on A40 |

Checkpoint folders: `/scratch1/mgordy/hingmmbert/12922539/hingmmbert/tok-{100,200,400,600,800,1000}M`. The intermediate checkpoints are mid-schedule (the learning rate is still higher than a standalone run of that length would end at). Present them as a training curve, not as "the 100M/200M model".

**V100s are unusable with our environment.** torch 2.11+cu128 `get_arch_list()` = sm_75, sm_80, sm_86, sm_90, sm_100, sm_120. Constrain jobs to `a40|l40s|a100` (P100 = sm_60 also out). The fp16 code path in `pretrain_mlm.py` stays (it worked locally) in case of a future torch build with sm_70.

**Pretraining throughput (measured, job 12919649):** mmBERT-base continued MLM, packed 512-token sequences, micro-batch 16 × 4 accumulation (64 seqs = 32,768 tokens/step), bf16, sparse MLM head: **26.4k tokens/s on an A40**, steady from minute 1 to 7. Peak GPU 28.9 GB of 44. MLM loss went 4.10 → 3.56 in the first 350 steps (still in warmup). The data pipeline alone does ~750k tokens/s, so the GPU is the bottleneck. Projected at 20 SU/min: 100M = 1.05 h / ~1.3k SU, 200M = 2.1 h / ~2.5k, **500M = 5.3 h / ~6.3k**, 1.2B = 12.6 h / ~15k, 2.4B (HingBERT-matched) = 25 h / ~30k. A100 not measured, expected ~1.5–2× faster at the same SU rate. Multi-hour runs belong on the `gpu` partition (debug caps jobs at 1 h): `--partition=gpu --gpus-per-task=1 --constraint="a40|l40s|a100"`.

The pool was ~141k / 600k SU used on 2026-10-10, up from 33k on 10-07, mostly other users. Usage decays with a 7-day half-life.

CARC lessons (also belong in `carc/carc_wiki.md`):
- `sbatch` from the **repo root**: the job runs in the submit directory, and the scripts use relative paths.
- Large pip/uv and model downloads stall on the login node. Use the transfer node `hpc-transfer1.usc.edu`, and set `UV_HTTP_TIMEOUT=600` and `UV_CONCURRENT_DOWNLOADS=2`.
- uv is in `~/.local/bin`; the caches are on `/scratch1/mgordy/{uv-cache,hf}`.
- `sbatch --test-only ...` shows when a job would start, without submitting it.
- The debug partition (1 h cap) often has a free A40 when the gpu partition is queued. Use it for short one-off runs, not sweeps.
