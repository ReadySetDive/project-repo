# Robustness of Hinglish Sentiment Classification to Real Spelling Variation

USC CSCI 544 project. Hinglish (Hindi written in Latin script, mixed with English) has no standard spelling: *accha*, *achha*, *acha*. We ask how much real, human-produced spelling variation hurts sentiment classifiers, and whether normalization helps.

1. **HingmmBERT**: repeat HingBERT's recipe (continued MLM pretraining on HingCorpus) on mmBERT-base.
2. **Attested-respelling test set**: perturb SentiMix with variants people actually wrote (hinglishNorm, Dakshina), varying degree and rarity.
3. **Normalization**: compare rule-based, lexicon-based and combined cleanup before classification.

Results and decisions so far: [NOTES.md](NOTES.md).

## Results so far (SentiMix test, macro-F1, single seed)

| Model | Frozen probe | Fine-tuned |
|---|---|---|
| HingBERT (110M) | 0.709 | **0.730** |
| F2LLM-v2-0.6B | 0.634 | 0.713 |
| mmBERT-base | 0.656 | 0.692 |
| Qwen3-0.6B-Base | n/a | 0.689 |

Seed variance is about ±1 F1, so only the HingBERT vs mmBERT gap is safe to claim. Details and significance tests are in NOTES.md.

## Layout

| Path | What |
|---|---|
| `embeddings/` | Modeling code: frozen MTEB probes (`run_mteb.py`), fine-tuning (`finetune.py`), MLM pretraining (`pretrain_mlm.py`), significance tests (`compare.py`). Small result files are in `results/`. |
| `embeddings/carc/` | Slurm scripts and environment setup for USC CARC |
| `carc/` | CARC sync script (`sync.sh`), notes, and the CARC Claude rules |
| `datasets/` | One folder per dataset, each with a `SOURCE.md` saying where it came from. Scripts for contamination checks, word overlap, and the hinglishNorm variant graph. |
| `llm/` | Claude-based labeling of SentiMix (language/dialect sounding) |

## Getting the data

Raw data is not in the repo (licenses differ and some files are large). Follow the `SOURCE.md` in each folder:

- `datasets/sentiment/sentimix/`: SentiMix (SemEval-2020 Task 9)
- `datasets/pretraining/hingcorpus/`: L3Cube-HingCorpus (roman), about 2.7 GB
- `datasets/perturbations/hinglishnorm/` and `dakshina-hi/`: respelling sources

## Running

Training runs on CARC (A40/L40S/A100; V100s are unsupported by our torch build).

```bash
wsl -d Ubuntu bash carc/sync.sh push        # send code and data to CARC (-n for a dry run)
sbatch embeddings/carc/finetune.sbatch      # submit from the repo root
wsl -d Ubuntu bash carc/sync.sh pull        # bring back results and logs
```

Environment setup is in `embeddings/carc/setup.sh`. CARC gotchas are in `carc/carc_wiki.md` and the end of NOTES.md.

Local scripts (contamination check, graph, labeling) are single files with usage in their docstrings; `llm/classify_sentimix.py` and similar declare their own dependencies inline (`uv run <script>`).

## Not tracked

Model checkpoints (multi-GB, on CARC scratch), raw datasets, Slurm logs, `.venv`, and local-only folders (`docs/`, `adjacent/`, `tools/`, `papers/`, `.vscode/`, `.claude/`). See `.gitignore`.
