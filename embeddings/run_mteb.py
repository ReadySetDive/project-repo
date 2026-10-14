# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mteb==2.24.4", "torch"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Linear-probe SentiMix with MTEB: frozen embeddings + logistic regression.

MTEB has no Hinglish sentiment task (its only hin-Latn tasks are the LinCE/PHINC
bitext ones), so this registers SentiMix as a custom AbsTaskClassification and
runs it through the normal mteb.evaluate path.

Models:
  codefuse-ai/F2LLM-v2-0.6B   in the MTEB registry (instruction prompt, last-token pooling)
  jhu-clsp/mmBERT-base        not in registry: loaded as plain SentenceTransformer (mean pooling)
  l3cube-pune/hing-bert       not in registry: same fallback

mmBERT and HingBERT are MLM encoders, not trained as sentence embedders, so their
frozen mean-pooled vectors understate what they do when fine-tuned. Compare them
with that in mind.

Tasks (--tasks):
  sentimix    SentiMix Hinglish (custom task, default)
  urdu-roman  UrduRomanSentimentClassification.v2 from MTEB (romanized Pakistani Urdu, UCI 2018)
              The only romanized task in MTEB(Indic). Cross-variety check, since SentiMix
              contains Roman Urdu too. Its labels come from a single source and are noisier.

Protocols (--protocol):
  mteb  MTEB's own: 10 runs, each trains LR on 32 tweets per label. Comparable to leaderboard-style numbers.
  full  1 run, LR trained on the whole train split. Closer to how we'd use the model.

Usage:
  uv run embeddings/run_mteb.py                                   all three models, full protocol
  uv run embeddings/run_mteb.py --tasks sentimix urdu-roman      both tasks
  uv run embeddings/run_mteb.py --models l3cube-pune/hing-bert --limit 300   smoke test
Results land in embeddings/results/<protocol>/ (mteb's own JSON layout) plus a printed table.
Per-item predictions go to predictions/, and gold labels to gold/, for compare.py.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import mteb
from datasets import Dataset, DatasetDict
from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.task_metadata import TaskMetadata

HERE = Path(__file__).resolve().parent
SENTIMIX = HERE.parent / "datasets" / "sentiment" / "sentimix"
MODELS = ["jhu-clsp/mmBERT-base", "codefuse-ai/F2LLM-v2-0.6B", "l3cube-pune/hing-bert"]
LABELS = ["negative", "neutral", "positive"]
MTEB_TASKS = {"urdu-roman": "UrduRomanSentimentClassification.v2"}

# reuse the CoNLL parser, mojibake fix and detokenizer from make_readable.py
_spec = importlib.util.spec_from_file_location("make_readable", SENTIMIX / "make_readable.py")
assert _spec and _spec.loader
readable = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readable)


def load_sentimix(limit: int | None) -> DatasetDict:
    test_labels = {}
    for line in (SENTIMIX / "test_labels_hinglish.txt").read_text(encoding="utf-8").splitlines()[1:]:
        uid, lab = line.split(",")
        test_labels[uid.strip()] = lab.strip()
    splits = {}
    for split, fname in readable.SPLITS.items():
        rows = {"id": [], "text": [], "label": []}
        for tid, lab, toks in readable.parse(SENTIMIX / fname):
            lab = lab or test_labels[tid]
            rows["id"].append(tid)
            rows["text"].append(readable.detok(toks))
            rows["label"].append(LABELS.index(lab))
        ds = Dataset.from_dict(rows)
        if limit:
            ds = ds.shuffle(seed=544).select(range(min(limit, len(ds))))
        splits[split] = ds
    return DatasetDict(splits)


class SentiMixHinglish(AbsTaskClassification):
    metadata = TaskMetadata(
        name="SentiMixHinglishClassification",
        description="Sentiment of Hindi-English code-mixed tweets (SemEval-2020 Task 9, SentiMix).",
        prompt="Classify the sentiment of a given Hinglish tweet as positive, negative, or neutral",
        reference="https://aclanthology.org/2020.semeval-1.100",
        dataset={"path": "local/sentimix", "revision": "local"},
        type="Classification",
        category="t2c",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["hin-Latn", "eng-Latn"],
        main_score="f1",                      # macro-F1, same as the SemEval task
        date=("2019-01-01", "2020-03-01"),
        domains=["Social", "Written"],
        task_subtypes=["Sentiment/Hate speech"],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )
    limit: int | None = None

    def load_data(self, num_proc=None, **kwargs):
        if self.data_loaded:
            return
        self.dataset = load_sentimix(self.limit)
        self.data_loaded = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tasks", nargs="+", default=["sentimix"], choices=["sentimix", *MTEB_TASKS])
    ap.add_argument("--protocol", choices=["mteb", "full"], default="full")
    ap.add_argument("--splits", nargs="+", default=["test"], help="SentiMix eval splits: test, dev")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, help="subsample every split to N tweets (smoke test)")
    args = ap.parse_args()

    tasks = []
    for key in args.tasks:
        if key == "sentimix":
            t = SentiMixHinglish()
            t.limit = args.limit
            t.metadata.eval_splits = args.splits
        else:
            t = mteb.get_task(MTEB_TASKS[key])
        if args.protocol == "full":
            t.samples_per_label = 10**9        # undersampling keeps everything
            t.n_experiments = 1
        else:
            t.samples_per_label = 32
            t.n_experiments = 10
        t.evaluator_model = t.evaluator_model.set_params(max_iter=1000)
        tasks.append(t)

    out = HERE / "results" / (args.protocol + (f"-limit{args.limit}" if args.limit else ""))
    table = []
    for name in args.models:
        model = mteb.get_model(name)         # registry entry, else plain SentenceTransformer(name)
        res = mteb.evaluate(model, tasks=tasks, encode_kwargs={"batch_size": args.batch_size},
                            cache=mteb.ResultCache(out), overwrite_strategy="always",
                            prediction_folder=out / "predictions" / name.replace("/", "__"))
        for tr in res.task_results:
            for split, subsets in tr.scores.items():
                s = subsets[0]
                table.append((name, tr.task_name, split, s["f1"], s["accuracy"]))

    # gold labels in the same order as the predictions, so compare.py needs no mteb
    (out / "gold").mkdir(parents=True, exist_ok=True)
    for t in tasks:
        t.load_data()                          # mteb.evaluate unloads each task when done
        ds = t.dataset["default"] if "default" in t.dataset else t.dataset
        for split in t.metadata.eval_splits:
            (out / "gold" / f"{t.metadata.name}_{split}.json").write_text(
                json.dumps(list(ds[split][t.label_column_name])), encoding="utf-8")

    print(f"\nprotocol={args.protocol}" + (f" limit={args.limit}" if args.limit else ""))
    print(f"{'model':28} {'task':36} {'split':6} {'macroF1':>8} {'acc':>7}")
    for name, task, split, f1, acc in table:
        print(f"{name:28} {task:36} {split:6} {f1:8.4f} {acc:7.4f}")


if __name__ == "__main__":
    main()
