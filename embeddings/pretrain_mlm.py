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
"""Continued masked-LM pretraining of an encoder on L3Cube-HingCorpus (HingmmBERT pilot).

Streams the corpus straight out of the .tar.gz (never extracted). The train file is already
shuffled, so its first N tokens are a random sample. Tweets are tokenized, joined with
[SEP] and packed into full 512-token sequences, so no compute goes to padding.

The corpus is used unfiltered, exactly as HingBERT saw it, so the two models get the same data.
Contamination is handled at evaluation time instead: score every model on the full test set
and on the clean subset from contamination.csv (see NOTES.md).

Settings follow HingBERT's recipe where it makes sense (batch 64 sequences, 15% masking) and
domain-adaptive pretraining practice elsewhere (lr 5e-5, 6% warmup). 15%, not mmBERT's 30%:
30% was only its first phase; the released model finished at 5% (arXiv 2509.06888), and
15% matches HingBERT, so the mask rate isn't a confound in the comparison.

--benchmark-minutes M stops after M minutes and prints measured tokens/s plus projected
hours and SU for each token budget. Same code path as the real run.

Usage:
  uv run embeddings/pretrain_mlm.py --corpus <HingCorpus .tar.gz> --benchmark-minutes 12
  uv run embeddings/pretrain_mlm.py --corpus <...> --tokens 500M --save-at 100M,200M,500M
  ... --stop-after-minutes 100     chunked: save a resumable checkpoint and exit; the same command resumes
Checkpoints go to $KEEP_DIR/<run>/tok-<N> (or results/pretrain/<run>/), each loadable by finetune.py.
Resumable trainer state (weights + optimizer + schedule + corpus position) goes to $CKPT_ROOT
(or results/pretrain/<run>/trainer/); DONE appears in $KEEP_DIR/<run>/ when the run is complete.
"""

import argparse
import json
import math
import os
import re
import shutil
import tarfile
import time
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
from transformers import (AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint

HERE = Path(__file__).resolve().parent
START = time.time()                               # --stop-after-minutes counts from process start (model load included)
SU_PER_MIN = 20                                   # 1 A40/A100/L40S + 4 CPU + 32 GB (carc_wiki.md section 7)

def parse_tokens(s: str) -> int:
    m = re.fullmatch(r"([\d.]+)\s*([kKmMbB]?)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad token count {s!r} (use e.g. 500M, 1.2B)")
    return int(float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6, "b": 1e9}[m.group(2).lower()])


def corpus_lines(path: Path, member: str, skip: int = 0):
    """Yields (line number, text). The first `skip` lines are read but not decoded (fast resume)."""
    with tarfile.open(path, "r:*") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(member):
                raw = tar.extractfile(m)
                assert raw
                for i, line in enumerate(raw):
                    if i >= skip:
                        yield i, line.decode("utf-8", errors="replace").strip()
                return
    raise SystemExit(f"{member} not found in {path}")


class PackedCorpus(IterableDataset):
    """Yields {"input_ids": [seq_len]} blocks of packed tweets.

    progress["lines"] is how many corpus lines have been consumed; it's recorded for each
    resumable checkpoint, and a resumed run skips that many lines (first pass only). Skipping
    loses at most the unyielded tail of one 1024-line batch plus the loader's one-batch
    lookahead, about one optimizer step of data; nothing is ever repeated.
    """

    def __init__(self, path, member, tok, seq_len, skip_lines=0, progress=None):
        self.path, self.member, self.tok, self.seq_len, self.skip = path, member, tok, seq_len, skip_lines
        self.progress = progress if progress is not None else {}

    def __iter__(self):
        buf, batch = [], []
        sep, cls = self.tok.sep_token_id, self.tok.cls_token_id
        skip, self.skip = self.skip, 0                    # if the corpus runs out, the next pass starts at line 0
        for i, line in corpus_lines(self.path, self.member, skip):
            if len(line) < 3:
                continue
            batch.append(line)
            if len(batch) < 1024:
                continue
            for ids in self.tok(batch, add_special_tokens=False)["input_ids"]:
                buf.extend(ids + [sep])
            batch = []
            self.progress["lines"] = i + 1
            while len(buf) >= self.seq_len - 1:
                yield {"input_ids": [cls] + buf[:self.seq_len - 1]}
                buf = buf[self.seq_len - 1:]


class Meter(TrainerCallback):
    """Logs tokens/s, saves at token milestones, stops at the benchmark or --stop-after-minutes
    limit, and records the corpus position next to each resumable checkpoint."""

    def __init__(self, tokens_per_step, args, model_out, tok, progress):
        self.tps, self.args, self.out, self.tok, self.progress = tokens_per_step, args, model_out, tok, progress
        self.t0 = self.warm_t = None
        self.warm_step = self.local_steps = 0
        self.save_at = sorted(args.save_at)
        self.result = {}

    def on_step_end(self, targs, state, control, **kw):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        self.local_steps += 1                             # steps in this process; global_step starts high on resume
        if self.local_steps == 20:                        # skip compile/warm-up steps in the rate
            self.warm_t, self.warm_step = now, state.global_step
        seen = state.global_step * self.tps
        while self.save_at and seen >= self.save_at[0]:
            mark = self.save_at.pop(0)
            d = self.out / f"tok-{mark // 1_000_000}M"
            if d.exists():                                # saved by an earlier job of this run; don't overwrite
                continue
            tmp = d.with_name(d.name + ".tmp")            # write, then rename: a kill mid-save never leaves a
            shutil.rmtree(tmp, ignore_errors=True)        # half-written tok-* that later chunks would skip
            kw["model"].save_pretrained(tmp)
            self.tok.save_pretrained(tmp)
            os.replace(tmp, d)
            print(f"saved {d} at step {state.global_step}", flush=True)
        if self.args.stop_after_minutes and now - START > self.args.stop_after_minutes * 60:
            print(f"time's up at step {state.global_step}: saving a resumable checkpoint and stopping", flush=True)
            control.should_save = True
            control.should_training_stop = True
        if control.should_save:                           # the Trainer saves checkpoint-<step> right after this;
            out = Path(targs.output_dir)                  # record the corpus position first, so it can't be missing
            out.mkdir(parents=True, exist_ok=True)
            (out / f"data_state-{state.global_step}.json").write_text(json.dumps(
                {"step": state.global_step, "lines": self.progress.get("lines", 0), "tokens": self.args.tokens}),
                encoding="utf-8")
        if state.global_step % 50 == 0 and self.warm_t and state.global_step > self.warm_step:
            rate = (state.global_step - self.warm_step) * self.tps / (now - self.warm_t)
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            self.result = {"step": state.global_step, "tokens_seen": seen, "tokens_per_s": rate,
                           "elapsed_min": (now - self.t0) / 60, "peak_gpu_gb": mem}
            print(json.dumps({k: round(v, 1) if isinstance(v, float) else v for k, v in self.result.items()}), flush=True)
        if self.args.benchmark_minutes and now - self.t0 > self.args.benchmark_minutes * 60:
            control.should_training_stop = True

    def on_save(self, targs, state, control, **kw):
        print(f"resumable checkpoint at step {state.global_step}, corpus line {self.progress.get('lines', 0):,}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--model", default="jhu-clsp/mmBERT-base")
    ap.add_argument("--tokens", type=parse_tokens, default=parse_tokens("500M"), help="training budget, e.g. 200M")
    ap.add_argument("--save-at", type=lambda s: [parse_tokens(x) for x in s.split(",")], default=[])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64, help="sequences per optimizer step (HingBERT: 64)")
    ap.add_argument("--micro-batch", type=int, help="sequences per forward pass; default 16 on >=40 GB GPUs, "
                    "8 on smaller ones (V100 32 GB). 32 OOMed on an A40 before the sparse-head fix")
    ap.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="auto: bf16 on Ampere+ (A40/L40S/A100), fp16 with loss scaling on V100")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--mlm-prob", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--run", default="hingmmbert")
    ap.add_argument("--benchmark-minutes", type=float, help="stop after this long and print projections")
    ap.add_argument("--stop-after-minutes", type=float,
                    help="save a resumable checkpoint and exit after this long; rerun the same command to continue")
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="optimizer steps between resumable checkpoints (~4 GB each, only the latest is kept)")
    args = ap.parse_args()
    set_seed(args.seed)
    cuda = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability() if cuda else (0, 0)
    # is_bf16_supported() can say yes on a V100 via slow emulation, so decide by compute capability
    prec = args.precision if args.precision != "auto" else ("bf16" if cap[0] >= 8 else "fp16" if cuda else "fp32")
    if args.micro_batch is None:
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30 if cuda else 0
        args.micro_batch = 16 if mem_gb >= 38 else 8          # an A100 "40 GB" reports 39.4 GiB
    print(f"gpu: {torch.cuda.get_device_name() if cuda else 'none'}, precision {prec}, micro-batch {args.micro_batch}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # Score only the masked tokens: with mmBERT's 256k vocab, full logits for 32x512 tokens are ~16 GB.
    # ModernBERT reads this flag once at construction (setting model.config afterwards does nothing;
    # that OOM'd job 12919395), so it has to go in as a from_pretrained override.
    extra = {"sparse_prediction": True} if "modernbert" in AutoConfig.from_pretrained(args.model).model_type else {}
    model = AutoModelForMaskedLM.from_pretrained(args.model, dtype=torch.float32, **extra)
    print(f"sparse MLM head: {getattr(model, 'sparse_prediction', 'n/a')}", flush=True)

    accum = max(1, args.batch // args.micro_batch)
    tokens_per_step = args.micro_batch * accum * args.seq_len
    steps = math.ceil(args.tokens / tokens_per_step)
    root = Path(os.environ.get("KEEP_DIR", HERE / "results" / "pretrain")) / args.run
    ckpt_dir = os.environ.get("CKPT_ROOT") or str(root / "trainer")

    # Resume: the Trainer restores weights, optimizer, LR schedule, RNG and step from the latest
    # checkpoint-<step>; the corpus position comes from data_state-<step>.json, written just before
    # that checkpoint (the Trainer's own data skipping would re-tokenize everything already seen).
    last = None if args.benchmark_minutes or not os.path.isdir(ckpt_dir) else get_last_checkpoint(ckpt_dir)
    skip = 0
    if last:
        state_file = Path(ckpt_dir) / f"data_state-{int(Path(last).name.split('-')[-1])}.json"
        if not state_file.exists():
            raise SystemExit(f"{state_file} missing, so the corpus position for {last} is unknown")
        st = json.loads(state_file.read_text(encoding="utf-8"))
        if st["tokens"] != args.tokens:
            raise SystemExit(f"{last} belongs to a {st['tokens']:,}-token run, this is {args.tokens:,}: "
                             "the LR schedule would not match. Use another --run or CKPT_ROOT.")
        skip = st["lines"]
        print(f"resuming from {last} (step {st['step']}), skipping {skip:,} corpus lines", flush=True)

    progress = {"lines": skip}
    train = PackedCorpus(args.corpus, "concatenated_train_final_shuffled.txt", tok, args.seq_len,
                         skip_lines=skip, progress=progress)
    print(f"{args.model}: {steps} steps x {tokens_per_step} tokens = {steps * tokens_per_step / 1e6:.0f}M tokens",
          flush=True)

    targs = TrainingArguments(
        output_dir=ckpt_dir, max_steps=steps,
        per_device_train_batch_size=args.micro_batch, gradient_accumulation_steps=accum,
        learning_rate=args.lr, warmup_steps=0.06, lr_scheduler_type="linear", weight_decay=0.01,
        bf16=prec == "bf16", fp16=prec == "fp16", logging_steps=50, report_to="none", seed=args.seed,
        save_strategy="no" if args.benchmark_minutes else "steps", save_steps=args.checkpoint_every,
        save_total_limit=1, ignore_data_skip=True,        # PackedCorpus skips by itself (above)
        dataloader_num_workers=0,                         # one streaming reader; tokenization keeps up
    )
    meter = Meter(tokens_per_step, args, root, tok, progress)
    trainer = Trainer(model=model, args=targs, train_dataset=train, callbacks=[meter],
                      data_collator=DataCollatorForLanguageModeling(tok, mlm_probability=args.mlm_prob))
    trainer.train(resume_from_checkpoint=last)

    if args.benchmark_minutes:
        r = meter.result
        if not r:
            raise SystemExit("too few steps to measure; run longer")
        print(f"\n== benchmark: {r['tokens_per_s']:,.0f} tokens/s, peak GPU {r['peak_gpu_gb']:.1f} GB")
        print(f"{'budget':>8} {'hours':>7} {'SU @' + str(SU_PER_MIN) + '/min':>12}")
        for b in ("100M", "200M", "500M", "1.2B", "2.4B"):
            h = parse_tokens(b) / r["tokens_per_s"] / 3600
            print(f"{b:>8} {h:7.1f} {h * 60 * SU_PER_MIN:12,.0f}")
        return

    if meter.local_steps == 0:                            # e.g. resumed past the end of the corpus: a chunk that
        raise SystemExit("no training steps ran; not continuing")  # exits 0 here would resubmit itself forever
    if trainer.state.global_step < steps:
        print(f"stopped at step {trainer.state.global_step}/{steps}; rerun the same command to continue", flush=True)
        return

    final = root / f"tok-{args.tokens // 1_000_000}M"
    if not final.exists():
        trainer.save_model(str(final))
        tok.save_pretrained(final)
    (root / "run.json").write_text(json.dumps({**vars(args), "corpus": str(args.corpus), "steps": steps,
                                               "tokens_per_step": tokens_per_step,
                                               "last": meter.result}, indent=2, default=str), encoding="utf-8")
    (root / "DONE").touch()                               # tells pretrain_pilot.sbatch not to queue another chunk
    print(f"done; checkpoints under {root}")


if __name__ == "__main__":
    main()
