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
"""Quick inference speed check: median ms per sentence, batch 1, on the first 100 SentiMix test
sentences (after 10 warmup), on GPU (bf16 autocast, like finetune.py's eval) and CPU (fp32).
Speed depends on the architecture, not the weights: mmBERT-base times the same as HingmmBERT.

Usage:
  uv run embeddings/infer_bench.py l3cube-pune/hing-bert jhu-clsp/mmBERT-base
  uv run embeddings/infer_bench.py --batches l3cube-pune/hing-bert jhu-clsp/mmBERT-base   GPU batch sweep
  uv run embeddings/infer_bench.py --long l3cube-pune/hing-bert jhu-clsp/mmBERT-base      same, ~450-token inputs
"""

import statistics
import sys
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import load_split

N = 100


def speed(model, enc, device):
    model.to(device)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    ms = []
    with torch.inference_mode(), ctx:
        for i, e in enumerate(enc):
            e = {k: v.to(device) for k, v in e.items()}
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            model(**e)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= 10:
                ms.append(1000 * (time.perf_counter() - t))
    return round(statistics.median(ms), 1)


def throughput(model, tok, texts, bs, by_length, max_len=128):
    """GPU sentences/s over all of `texts` at batch size bs, tokenization excluded. Sorting by
    length first means less padding per batch, a cheap stand-in for ModernBERT's unpadding."""
    if by_length:
        texts = sorted(texts, key=len)
    batches = [tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
               for i in range(0, len(texts), bs)]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(**batches[0])                                   # warmup
        torch.cuda.synchronize()
        t = time.perf_counter()
        for b in batches:
            model(**b)
        torch.cuda.synchronize()
    return round(len(texts) / (time.perf_counter() - t))


def main():
    if "--long" in sys.argv:                              # near BERT's 512-token limit: 10 test sentences per input
        sents = load_split("test")["text"]
        texts = [" ".join(sents[i:i + 10]) for i in range(0, len(sents), 10)]
        print(f"{len(texts)} inputs of 10 sentences; {torch.cuda.get_device_name()}; inputs/s", flush=True)
        for name in [a for a in sys.argv[1:] if a != "--long"]:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).eval().to("cuda")
            toks = statistics.mean(len(tok(t, truncation=True, max_length=512)["input_ids"]) for t in texts)
            res = ", ".join(f"bs{bs} {throughput(model, tok, texts, bs, False, 512)}" for bs in (1, 4, 8, 16, 32, 64))
            print(f"{name} [{model.config._attn_implementation}, {toks:.0f} tokens/input]: {res}", flush=True)
        return
    if "--batches" in sys.argv:                           # GPU batch-size sweep on the whole test set
        texts = load_split("test")["text"]
        print(f"{len(texts)} sentences; {torch.cuda.get_device_name()}; sentences/s", flush=True)
        for name in [a for a in sys.argv[1:] if a != "--batches"]:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3).eval().to("cuda")
            for by_length in (False, True):
                res = ", ".join(f"bs{bs} {throughput(model, tok, texts, bs, by_length)}" for bs in (8, 32, 128, 512))
                print(f"{name} [{model.config._attn_implementation}, {'length-sorted' if by_length else 'file order'}]: "
                      f"{res}", flush=True)
        return
    texts = load_split("test")["text"][:N + 10]
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    print(f"{N} sentences, batch 1; {torch.cuda.get_device_name() if 'cuda' in devices else 'no GPU'}; "
          f"{torch.get_num_threads()} CPU threads", flush=True)
    for name in sys.argv[1:]:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, dtype=torch.float32).eval()
        enc = [tok(t, truncation=True, max_length=128, return_tensors="pt") for t in texts]
        params = sum(p.numel() for p in model.parameters()) / 1e6
        toks = statistics.mean(e["input_ids"].shape[1] for e in enc)
        times = ", ".join(f"{d} {speed(model, enc, d)} ms" for d in devices)
        print(f"{name}: {params:.0f}M params, {toks:.1f} tokens/sentence, {times}", flush=True)


if __name__ == "__main__":
    main()
