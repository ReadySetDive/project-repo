# /// script
# requires-python = ">=3.10"
# dependencies = ["anthropic"]
# ///
"""Label SentiMix tweets with Claude. Two tasks, same resumable batch machinery:

  --task sounding   open-ended: which languages / varieties / scripts appear (no fixed list),
                    to discover what's actually in the data (Tagalog, Punjabi, ...)
  --task main       sentiment + dialect from a fixed list; choosing "other" requires naming
                    the variety in `other_dialect`, so new ones surface and can be deduped

Default model is Claude Haiku 5.5 (cheap); ramp with --model claude-sonnet-5-5 / claude-opus-5-5.
Each run lives in llm/runs/<run>/ with a state.json, so you can submit, close the terminal,
and come back later. The task and model are saved in state; later commands reuse them.

Usage (from the repo root):
    uv run llm/classify_sentimix.py submit  --run sound-v1 --task sounding --split all --n 1000 --dry-run
    uv run llm/classify_sentimix.py submit  --run sound-v1 --task sounding --split all --n 1000
    uv run llm/classify_sentimix.py status  --run sound-v1
    uv run llm/classify_sentimix.py collect --run sound-v1                      # results.csv + summary.md
    uv run llm/classify_sentimix.py retry   --run sound-v1 [--model claude-opus-5-5]   # refused/failed rows
    uv run llm/classify_sentimix.py run     --run main-test-v1 --task main --split test   # submit + wait + collect

Run folder:
    state.json            task, model, settings, batch id, status history
    sample.csv            the tweets sent (id, split, sentiment, phrase)
    request_example.json  one full request, for inspection
    results.jsonl         one raw record per tweet (parsed output, usage, stop reason)
    results.csv           flattened results
    summary.md            sounding: counts of free-text varieties / languages / scripts / cues
                          main: sentiment vs gold (overall + per dialect), dialect counts,
                          every `other` label (dedupe list), schema violations
                          both: refusals, measured tokens and cost

Needs the readable CSVs: python datasets/sentiment/sentimix/make_readable.py
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
READABLE = REPO / "datasets" / "sentiment" / "sentimix" / "readable"
RUNS = Path(__file__).resolve().parent / "runs"
DEFAULT_MODEL = "claude-haiku-5-5"

# $ per million tokens (input, output), standard API rates; Batch API is 50% of these.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Models that accept server-side refusal fallback (`fallbacks: "default"`); Haiku 5.5 has none.
FALLBACK_OK = {"claude-opus-5-5", "claude-sonnet-5-5"}

LABELS = ["positive", "negative", "neutral"]
DIALECTS = ["indian_hinglish", "pakistani_roman_urdu", "english", "unclear", "other"]

# ---------------------------------------------------------------- task: sounding

SOUNDING_SYSTEM = """You are a careful sociolinguist who studies multilingual social media text.

For the post you are given, describe which languages and varieties it is written in. Do not force it into a fixed set of categories: name what you actually see. The collection is mostly South Asian code-mixed text in the Latin alphabet, but posts may be in any language, any mix of languages, or any script. If the evidence is thin (a very short or generic post), say so with low confidence rather than guessing.

Field guidance:
- languages: every language present, with its role. "matrix" is the language supplying the grammatical frame, "embedded" means words or phrases inserted into it, "minor" means a stray word or name.
- variety: a short free-text name for the regional or national variety, e.g. "Indian Hinglish", "Pakistani Roman Urdu", "Punjabi-English", "Indian English", "Tagalog", "unclear".
- scripts: the writing systems that appear, e.g. "Latin", "Devanagari", "Gurmukhi", "Arabic", "emoji only".
- cues: the specific words or spellings that support your call, copied exactly from the post.
- confidence: how sure you are of the variety.
- notes: one short sentence on anything unusual (e.g. "mostly hashtags", "could be either Hindi or Urdu"). Keep it brief."""

SOUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "languages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "role": {"type": "string", "enum": ["matrix", "embedded", "minor"]},
                },
                "required": ["language", "role"],
                "additionalProperties": False,
            },
        },
        "variety": {"type": "string"},
        "scripts": {"type": "array", "items": {"type": "string"}},
        "cues": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "notes": {"type": "string"},
    },
    "required": ["languages", "variety", "scripts", "cues", "confidence", "notes"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------- task: main

MAIN_SYSTEM = """You label social media posts, mostly code-mixed Hindi/Urdu-English written in the Latin alphabet.

For each post, give:

sentiment: the overall sentiment the author expresses.
- positive: approval, praise, congratulations, affection, happiness, support.
- negative: criticism, anger, insults, sadness, complaint, sarcasm whose intent is negative.
- neutral: factual, informational, questions, or no clear sentiment.
If the post mixes polarities, choose the one that dominates the author's overall message.

dialect: the variety the post is written in.
- indian_hinglish: Hindi-English mixing with Indian romanization habits (e.g. "hai", "mein", "kya", "nahi").
- pakistani_roman_urdu: Urdu-English mixing with Pakistani romanization habits (e.g. "hy", "ky", "-ay" endings like "hotay", "ghalat", "kia").
- english: entirely or almost entirely English.
- unclear: too short or generic to tell (e.g. only names, hashtags, emoji, or words shared by both varieties).
- other: anything else, such as Punjabi, Bhojpuri, Marathi, Bengali, Tagalog, any other language, or a non-Latin script.

other_dialect: if and only if dialect is "other", name the language or variety in a few words (e.g. "Punjabi (Gurmukhi script)", "Tagalog"). Otherwise use an empty string.

dialect_cues: the specific words or spellings that support the dialect choice, copied exactly from the post. Use an empty list when there are none.

explanation: one short sentence naming the words or cues that decided the sentiment label."""

MAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": LABELS},
        "dialect": {"type": "string", "enum": DIALECTS},
        "other_dialect": {"type": "string"},
        "dialect_cues": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["sentiment", "dialect", "other_dialect", "dialect_cues", "explanation"],
    "additionalProperties": False,
}

TASKS = {
    "sounding": {"system": SOUNDING_SYSTEM, "schema": SOUNDING_SCHEMA, "est_in": 450},
    "main": {"system": MAIN_SYSTEM, "schema": MAIN_SCHEMA, "est_in": 650},
}


# ---------------------------------------------------------------- state + io

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_dir(run: str) -> Path:
    return RUNS / run


def load_state(run: str) -> dict:
    p = run_dir(run) / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(run: str, state: dict):
    d = run_dir(run)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / "state.json")  # atomic, so an interrupted write never corrupts state


def log(state: dict, event: str, **kw):
    state.setdefault("history", []).append({"at": now(), "event": event, **kw})


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], cols: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def client():
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic SDK not installed: run with `uv run llm/classify_sentimix.py ...` or `pip install anthropic`")
    return anthropic.Anthropic()


# ---------------------------------------------------------------- sampling + requests

def load_rows(split: str, n: int | None, seed: int) -> list[dict]:
    splits = ["train", "dev", "test"] if split == "all" else [split]
    rows = []
    for s in splits:
        p = READABLE / f"{s}.csv"
        if not p.exists():
            sys.exit(f"{p} missing: run python datasets/sentiment/sentimix/make_readable.py first")
        rows += [{**r, "split": s} for r in read_csv(p)]
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def params(task: str, model: str, effort: str, phrase: str) -> dict:
    t = TASKS[task]
    return {
        "model": model,
        # thinking is on by default on current models and counts toward max_tokens; leave room for it
        "max_tokens": 4000,
        "system": t["system"],
        "messages": [{"role": "user", "content": f"Post:\n{phrase}"}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": t["schema"]}},
    }


def custom_id(row: dict) -> str:
    return f"{row['split']}-{row['id']}"


# ---------------------------------------------------------------- commands

def cmd_submit(a):
    state = load_state(a.run)
    if state.get("batch_id") and not a.force:
        print(f"run '{a.run}' already submitted as {state['batch_id']} (use status/collect, or --force to resubmit)")
        return state
    if state.get("task") and state["task"] != a.task and not a.force:
        sys.exit(f"run '{a.run}' was created as task '{state['task']}'; pick a new --run name")
    d = run_dir(a.run)
    d.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.split, a.n, a.seed)
    write_csv(d / "sample.csv", rows, ["id", "split", "sentiment", "phrase"])
    example = params(a.task, a.model, a.effort, rows[0]["phrase"])
    (d / "request_example.json").write_text(json.dumps(example, indent=2, ensure_ascii=False), encoding="utf-8")
    state.update({"run": a.run, "task": a.task, "model": a.model, "effort": a.effort, "split": a.split,
                  "n": len(rows), "seed": a.seed, "created": state.get("created", now())})
    if a.model in PRICES:
        pin, pout = PRICES[a.model]
        est_in = TASKS[a.task]["est_in"]
        # rough: system + schema + post in; JSON + low-effort thinking out
        lo = len(rows) * (est_in * pin + 150 * pout) / 1e6 / 2
        hi = len(rows) * (est_in * pin + 800 * pout) / 1e6 / 2
        print(f"estimated batch cost for {len(rows)} posts, task {a.task}, {a.model}: ${lo:.2f} - ${hi:.2f}")
    if a.dry_run:
        log(state, "dry_run", n=len(rows))
        save_state(a.run, state)
        print(f"dry run: wrote sample.csv + request_example.json to {d}; nothing sent")
        return state

    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    reqs = [Request(custom_id=custom_id(r),
                    params=MessageCreateParamsNonStreaming(**params(a.task, a.model, a.effort, r["phrase"])))
            for r in rows]
    batch = client().messages.batches.create(requests=reqs)
    state.update({"batch_id": batch.id, "status": batch.processing_status})
    log(state, "submitted", batch_id=batch.id, n=len(reqs))
    save_state(a.run, state)
    print(f"submitted batch {batch.id} ({len(reqs)} requests); state saved to {d / 'state.json'}")
    return state


def cmd_status(a, quiet=False):
    state = load_state(a.run)
    if not state.get("batch_id"):
        sys.exit(f"run '{a.run}' has no submitted batch")
    b = client().messages.batches.retrieve(state["batch_id"])
    c = b.request_counts
    counts = {"processing": c.processing, "succeeded": c.succeeded, "errored": c.errored,
              "canceled": c.canceled, "expired": c.expired}
    if b.processing_status != state.get("status") or counts != state.get("counts"):
        log(state, "status", status=b.processing_status, **counts)
    state.update({"status": b.processing_status, "counts": counts, "checked": now()})
    save_state(a.run, state)
    if not quiet:
        print(f"{state['batch_id']}: {b.processing_status}  {counts}")
    return state


def parse_message(msg) -> tuple[dict | None, str]:
    """Return (parsed JSON or None, stop_reason)."""
    if msg.stop_reason == "refusal":
        return None, "refusal"
    text = next((blk.text for blk in msg.content if blk.type == "text"), "")
    try:
        return json.loads(text), msg.stop_reason
    except json.JSONDecodeError:
        return None, f"{msg.stop_reason}:bad_json"


def record(cid: str, row: dict, msg=None, error: str = "", via: str = "batch") -> dict:
    rec = {"custom_id": cid, "id": row["id"], "split": row["split"], "gold": row.get("sentiment", row.get("gold", "")),
           "phrase": row["phrase"], "via": via, "error": error, "output": None, "stop_reason": "",
           "model": "", "input_tokens": 0, "output_tokens": 0}
    if msg is not None:
        out, stop = parse_message(msg)
        rec.update({"output": out, "stop_reason": stop, "model": msg.model,
                    "input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens})
        if stop == "refusal" and getattr(msg, "stop_details", None):
            rec["error"] = f"refusal:{msg.stop_details.category}"
    return rec


def cmd_collect(a):
    state = cmd_status(a, quiet=True)
    if state["status"] != "ended":
        print(f"batch not finished yet: {state['status']} {state['counts']}")
        return state
    d = run_dir(a.run)
    rows = {custom_id(r): r for r in read_csv(d / "sample.csv")}
    recs = {}
    for res in client().messages.batches.results(state["batch_id"]):
        row = rows.get(res.custom_id)
        if row is None:
            continue
        t = res.result.type
        if t == "succeeded":
            recs[res.custom_id] = record(res.custom_id, row, res.result.message)
        else:
            err = getattr(getattr(res.result, "error", None), "type", "") if t == "errored" else ""
            recs[res.custom_id] = record(res.custom_id, row, error=f"{t}{':' + err if err else ''}")
    # keep any rows already fixed by an earlier `retry`
    old = d / "results.jsonl"
    if old.exists():
        for line in old.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("via", "").startswith("retry") and r.get("output"):
                recs[r["custom_id"]] = r
    write_results(state["task"], d, list(recs.values()))
    log(state, "collected", n=len(recs))
    save_state(a.run, state)
    summarize(a.run, state, list(recs.values()))
    return state


def cmd_retry(a):
    """Re-run refused / errored / unparseable rows one at a time, at standard (non-batch) rates.
    Uses server-side fallback where the model supports it (not Haiku 5.5, and never inside a Batch).
    --model escalates the retries to a different model, e.g. claude-opus-5-5."""
    state = load_state(a.run)
    d = run_dir(a.run)
    path = d / "results.jsonl"
    if not path.exists():
        sys.exit("no results yet: run collect first")
    model = a.model or state["model"]
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    todo = [r for r in recs if not r.get("output")]
    print(f"{len(todo)} rows to retry on {model}")
    c = client()
    for r in todo:
        p = params(state["task"], model, state["effort"], r["phrase"])
        try:
            if model in FALLBACK_OK:
                msg = c.beta.messages.create(**p, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                msg = c.messages.create(**p)
            new = record(r["custom_id"], r, msg, via=f"retry:{model}")
        except Exception as e:  # keep going; the row stays failed and is reported
            new = {**r, "via": f"retry:{model}", "error": f"retry_failed:{type(e).__name__}: {e}"}
        r.clear()
        r.update(new)
        print(f"  {r['custom_id']}: {'ok' if r.get('output') else r.get('error') or r.get('stop_reason')}")
    write_results(state["task"], d, recs)
    log(state, "retried", model=model, n=len(todo), fixed=sum(1 for r in todo if r.get("output")))
    save_state(a.run, state)
    summarize(a.run, state, recs)


def cmd_run(a):
    state = load_state(a.run)
    if not state.get("batch_id"):
        state = cmd_submit(a)
        if a.dry_run:
            return
    while True:
        state = cmd_status(a)
        if state["status"] == "ended":
            break
        time.sleep(a.poll)
    cmd_collect(a)


# ---------------------------------------------------------------- outputs

def violation(o: dict) -> str:
    """Main task rule the schema can't enforce: other_dialect filled iff dialect == other."""
    od = o.get("other_dialect", "").strip()
    if o.get("dialect") == "other" and not od:
        return "other_without_label"
    if o.get("dialect") != "other" and od:
        return "label_without_other"
    return ""


def flat_row(task: str, r: dict) -> dict:
    o = r.get("output") or {}
    base = {"id": r["id"], "split": r["split"], "gold": r["gold"]}
    tail = {"phrase": r["phrase"], "stop_reason": r.get("stop_reason", ""), "error": r.get("error", ""),
            "via": r.get("via", ""), "model": r.get("model", "")}
    if task == "sounding":
        return {**base,
                "variety": o.get("variety", ""), "confidence": o.get("confidence", ""),
                "languages": "; ".join(f"{x['language']} ({x['role']})" for x in o.get("languages", [])),
                "scripts": "; ".join(o.get("scripts", [])), "cues": "; ".join(o.get("cues", [])),
                "notes": o.get("notes", ""), **tail}
    return {**base, "pred": o.get("sentiment", ""), "correct": (o.get("sentiment") == r["gold"]) if o else "",
            "dialect": o.get("dialect", ""), "other_dialect": o.get("other_dialect", ""),
            "dialect_cues": "; ".join(o.get("dialect_cues", [])), "explanation": o.get("explanation", ""),
            "violation": violation(o) if o else "", **tail}


def write_results(task: str, d: Path, recs: list[dict]):
    with (d / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    flat = [flat_row(task, r) for r in recs]
    write_csv(d / "results.csv", flat, list(flat[0].keys()) if flat else ["id"])


def scores(pairs: list[tuple[str, str]]) -> dict:
    """Accuracy, macro-F1 and per-class F1 over (gold, pred) pairs."""
    n = len(pairs)
    if not n:
        return {"n": 0, "acc": 0.0, "macro_f1": 0.0, "f1": {}}
    f1 = {}
    for lab in LABELS:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": n, "acc": sum(g == p for g, p in pairs) / n, "macro_f1": sum(f1.values()) / len(LABELS), "f1": f1}


def count_table(c: Counter, total: int, header: str, k: int = 50) -> list[str]:
    rows = [f"| {name or '(blank)'} | {n} | {n / max(total, 1):.1%} |" for name, n in c.most_common(k)]
    return [f"| {header} | n | share |", "|---|---|---|"] + (rows or ["| (none) | | |"])


def usage_line(state: dict, recs: list[dict]) -> str:
    tin = sum(r.get("input_tokens", 0) for r in recs)
    tout = sum(r.get("output_tokens", 0) for r in recs)
    cost = 0.0
    for r in recs:  # price each row by the model that served it; batch rows at 50%
        run_model = state.get("model") or ""
        pin, pout = PRICES.get(r.get("model") or run_model, PRICES.get(run_model, (0, 0)))
        mult = 0.5 if r.get("via") == "batch" else 1.0
        cost += (r.get("input_tokens", 0) * pin + r.get("output_tokens", 0) * pout) / 1e6 * mult
    return f"tokens in {tin:,} / out {tout:,} · measured cost ~${cost:.2f}"


def summarize(run: str, state: dict, recs: list[dict]):
    task = state["task"]
    ok = [r for r in recs if r.get("output")]
    bad = [r for r in recs if not r.get("output")]
    head = [f"# {run} ({task})", "",
            f"model `{state.get('model')}` · effort `{state.get('effort')}` · {state.get('n')} posts from "
            f"`{state.get('split')}` (seed {state.get('seed')})",
            f"parsed {len(ok)} · failed {len(bad)} · {usage_line(state, recs)}", ""]
    if task == "sounding":
        body, console = summarize_sounding(ok)
    else:
        body, console = summarize_main(ok)
    fail = ["", "## Failed rows", ""] + (
        [f"- `{r['custom_id']}`: {r.get('error') or r.get('stop_reason')}" for r in bad] or ["none"])
    (run_dir(run) / "summary.md").write_text("\n".join(head + body + fail) + "\n", encoding="utf-8")
    print(f"parsed {len(ok)}, failed {len(bad)}, {usage_line(state, recs)}")
    for line in console:
        print(line)
    print(f"wrote results.csv + summary.md to {run_dir(run)}")


def summarize_sounding(ok: list[dict]):
    n = len(ok)
    variety = Counter(r["output"]["variety"].strip() for r in ok)
    variety_hi = Counter(r["output"]["variety"].strip() for r in ok if r["output"]["confidence"] == "high")
    matrix = Counter(x["language"].strip() for r in ok for x in r["output"]["languages"] if x["role"] == "matrix")
    anylang = Counter(x["language"].strip() for r in ok for x in r["output"]["languages"])
    scripts = Counter(s.strip() for r in ok for s in r["output"]["scripts"])
    conf = Counter(r["output"]["confidence"] for r in ok)
    cues = Counter(c.strip().lower() for r in ok for c in r["output"]["cues"])
    body = (["## Variety (free text, as the model wrote it)", ""] + count_table(variety, n, "variety", 80)
            + ["", "## Variety, high-confidence only", ""] + count_table(variety_hi, n, "variety", 80)
            + ["", "## Matrix language", ""] + count_table(matrix, n, "language")
            + ["", "## Any language present", ""] + count_table(anylang, n, "language")
            + ["", "## Scripts", ""] + count_table(scripts, n, "script")
            + ["", "## Confidence", ""] + count_table(conf, n, "confidence")
            + ["", "## Most common cues", ""] + count_table(cues, n, "cue", 80))
    console = ["varieties: " + ", ".join(f"{k} ({v})" for k, v in variety.most_common(12)),
               "languages present: " + ", ".join(f"{k} ({v})" for k, v in anylang.most_common(15))]
    return body, console


def summarize_main(ok: list[dict]):
    graded = [r for r in ok if r["gold"] in LABELS]
    overall = scores([(r["gold"], r["output"]["sentiment"]) for r in graded])
    by_dialect = defaultdict(list)
    for r in graded:
        by_dialect[r["output"]["dialect"]].append((r["gold"], r["output"]["sentiment"]))
    confusion = Counter((r["gold"], r["output"]["sentiment"]) for r in graded)
    dialects = Counter(r["output"]["dialect"] for r in ok)
    others = Counter(r["output"]["other_dialect"].strip().lower() for r in ok if r["output"]["dialect"] == "other")
    violations = [(r, violation(r["output"])) for r in ok if violation(r["output"])]
    cues = defaultdict(Counter)
    for r in ok:
        for c in r["output"]["dialect_cues"]:
            cues[r["output"]["dialect"]][c.strip().lower()] += 1

    def score_row(name, s):
        f = s["f1"]
        return (f"| {name} | {s['n']} | {s['acc']:.3f} | {s['macro_f1']:.3f} | "
                + " | ".join(f"{f.get(l, 0):.3f}" for l in LABELS) + " |")

    body = ["## Sentiment vs gold", "",
            "| subset | n | accuracy | macro-F1 | F1 pos | F1 neg | F1 neu |", "|---|---|---|---|---|---|---|",
            score_row("all", overall)]
    body += [score_row(f"dialect: {k}", scores(v)) for k, v in sorted(by_dialect.items(), key=lambda kv: -len(kv[1]))]
    body += ["", "Confusion (rows = gold, columns = predicted):", "",
             "| gold \\ pred | " + " | ".join(LABELS) + " |", "|---" * (len(LABELS) + 1) + "|"]
    body += [f"| {g} | " + " | ".join(str(confusion[(g, p)]) for p in LABELS) + " |" for g in LABELS]
    body += ["", "## Dialect", ""] + count_table(dialects, len(ok), "dialect")
    body += ["", "## `other` labels (dedupe these)", ""] + count_table(others, len(ok), "other_dialect", 200)
    body += ["", "## Top dialect cues per dialect", ""]
    for k, c in sorted(cues.items(), key=lambda kv: -sum(kv[1].values())):
        body.append(f"- **{k}**: " + ", ".join(f"`{w}` ({n})" for w, n in c.most_common(25)))
    body += ["", f"## Schema violations ({len(violations)})", ""]
    body += [f"- `{r['custom_id']}`: {v} (dialect={r['output']['dialect']}, other_dialect={r['output']['other_dialect']!r})"
             for r, v in violations] or ["none"]
    console = [f"sentiment: acc {overall['acc']:.3f}, macro-F1 {overall['macro_f1']:.3f} (n={overall['n']})",
               "dialects: " + ", ".join(f"{k} ({v})" for k, v in dialects.most_common()),
               f"schema violations: {len(violations)}"]
    if others:
        console.append("other: " + ", ".join(f"{k} ({v})" for k, v in others.most_common(12)))
    return body, console


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("submit", "status", "collect", "retry", "run"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, help="run name; state lives in llm/runs/<run>/")
        if name in ("submit", "run"):
            p.add_argument("--task", default="main", choices=list(TASKS))
            p.add_argument("--split", default="test", choices=["all", "train", "dev", "test"])
            p.add_argument("--n", type=int, help="random subset of this size (default: the whole split)")
            p.add_argument("--seed", type=int, default=544)
            p.add_argument("--model", default=DEFAULT_MODEL)
            p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
            p.add_argument("--dry-run", action="store_true", help="sample + preview only, no API call")
            p.add_argument("--force", action="store_true", help="resubmit even if this run already has a batch")
        if name == "retry":
            p.add_argument("--model", help="escalate retries to this model (default: the run's model)")
        if name == "run":
            p.add_argument("--poll", type=int, default=60, help="seconds between status checks")
    a = ap.parse_args()
    {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect, "retry": cmd_retry, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
