# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["transformers>=4.56", "torch", "accelerate", "datasets", "scikit-learn", "numpy"]
#
# [tool.uv.sources]
# torch = { index = "pytorch-cu128" }
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""Fine-tune a model end to end for SentiMix 3-way sentiment, the fair version of run_mteb.py.

Every model gets a fresh classification head (AutoModelForSequenceClassification:
[CLS] head for BERT/ModernBERT, last-token head for Qwen3/F2LLM) and full fine-tuning.
The checkpoint with the best dev macro-F1 is scored once on test.

Predictions are written in the same layout as run_mteb.py, so compare.py works on them:
  results/finetune[-<tag>]/predictions/<model>/SentiMixHinglishClassification_predictions.json
  results/finetune[-<tag>]/gold/SentiMixHinglishClassification_test.json

Usage:
  uv run embeddings/finetune.py --model l3cube-pune/hing-bert
  uv run embeddings/finetune.py --model jhu-clsp/mmBERT-base --lr 5e-5
  uv run embeddings/finetune.py --model l3cube-pune/hing-bert --limit 500 --epochs 1   smoke test
  uv run embeddings/compare.py --protocol finetune
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

HERE = Path(__file__).resolve().parent
SENTIMIX = HERE.parent / "datasets" / "sentiment" / "sentimix"
LABELS = ["negative", "neutral", "positive"]   # must match run_mteb.py / compare.py
TASK = "SentiMixHinglishClassification"

# learning rates that are standard for each family; override with --lr
DEFAULT_LR = {"l3cube-pune/hing-bert": 2e-5, "jhu-clsp/mmBERT-base": 5e-5, "codefuse-ai/F2LLM-v2-0.6B": 1e-5,
              "Qwen/Qwen3-0.6B-Base": 1e-5}           # same as F2LLM, its fine-tuned descendant

_spec = importlib.util.spec_from_file_location("make_readable", SENTIMIX / "make_readable.py")
assert _spec and _spec.loader
readable = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readable)


def load_split(split: str) -> dict:
    test_labels = {}
    for line in (SENTIMIX / "test_labels_hinglish.txt").read_text(encoding="utf-8").splitlines()[1:]:
        uid, lab = line.split(",")
        test_labels[uid.strip()] = lab.strip()
    rows = {"text": [], "label": []}
    for tid, lab, toks in readable.parse(SENTIMIX / readable.SPLITS[split]):
        rows["text"].append(readable.detok(toks))
        rows["label"].append(LABELS.index(lab or test_labels[tid]))
    return rows


def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro"), "accuracy": accuracy_score(labels, preds)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--tag", default="", help="suffix for the results dir, e.g. a seed or lr sweep")
    ap.add_argument("--limit", type=int, help="subsample train/dev/test to N (smoke test)")
    args = ap.parse_args()
    lr = args.lr or DEFAULT_LR.get(args.model, 2e-5)
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}, dtype=torch.float32)   # fp32 master weights; bf16 autocast below
    model.config.pad_token_id = tok.pad_token_id   # Qwen3 head finds the last non-pad token with this

    def encode(split):
        ds = Dataset.from_dict(load_split(split))
        if args.limit:
            ds = ds.shuffle(seed=544).select(range(min(args.limit, len(ds))))
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True, remove_columns=["text"])
    train, dev, test = encode("train"), encode("dev"), encode("test")

    name = args.model.replace("/", "__")
    out = HERE / "results" / ("finetune" + (f"-{args.tag}" if args.tag else "")
                              + (f"-limit{args.limit}" if args.limit else ""))
    # CKPT_ROOT puts the (deleted-at-the-end) checkpoints on scratch instead of next to the results
    ckpt = Path(os.environ.get("CKPT_ROOT", out / "checkpoints")) / name
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=str(ckpt), learning_rate=lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=128,
        warmup_steps=0.1,                     # <1 means a fraction of total steps (transformers 5)
        weight_decay=0.01, lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        bf16=bf16, seed=args.seed, logging_steps=50, report_to="none", dataloader_num_workers=2,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    t0 = time.time()
    trainer.train()
    train_min = (time.time() - t0) / 60

    dev_m = trainer.evaluate(dev, metric_key_prefix="dev")
    pred = trainer.predict(test, metric_key_prefix="test")
    preds = np.argmax(pred.predictions, axis=-1).tolist()

    (out / "predictions" / name).mkdir(parents=True, exist_ok=True)
    (out / "predictions" / name / f"{TASK}_predictions.json").write_text(json.dumps(
        {"mteb_model_meta": {"model_name": args.model}, "default": {"test": [preds]}}), encoding="utf-8")
    (out / "gold").mkdir(parents=True, exist_ok=True)
    (out / "gold" / f"{TASK}_test.json").write_text(json.dumps(list(test["label"])), encoding="utf-8")
    summary = {"model": args.model, "lr": lr, "epochs": args.epochs, "batch_size": args.batch_size,
               "seed": args.seed, "train_minutes": round(train_min, 1),
               "dev_macro_f1": dev_m["dev_macro_f1"], "test_macro_f1": pred.metrics["test_macro_f1"],
               "test_accuracy": pred.metrics["test_accuracy"]}

    # KEEP_DIR saves the best model (weights + tokenizer, no optimizer state) so later test sets
    # (e.g. attested respellings) are scored with this exact model, not a retrained one.
    if os.environ.get("KEEP_DIR"):
        keep = Path(os.environ["KEEP_DIR"]) / name
        trainer.save_model(str(keep))
        tok.save_pretrained(str(keep))
        (keep / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["saved_to"] = str(keep)

    (out / "predictions" / name / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    import shutil                                   # epoch checkpoints carry optimizer state (2-3x the model size)
    shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    main()
