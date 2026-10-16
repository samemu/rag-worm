#!/usr/bin/env python3
"""E1: was the earlier (pre-fix) configuration prompt silently truncated?

Rebuilds the earlier (pre-fix) configuration prompt shape (top_k=10 abstracts + 5 'recent responses' + the permissive
clause, all in the user message) and plants two sentinels:
  REF code word : at the very START of the reference context (lost first if truncated from the front)
  SYS code word : inside the system prompt
The model is asked to report both. Each prompt is sent via
  (a) legacy path : /v1/chat/completions (what the earlier (pre-fix) configuration used; num_ctx cannot be set there)
  (b) native path : /api/chat with options.num_ctx (what v2 uses)
and we record reported prompt tokens vs. an estimate of the true prompt size.

  python scripts/e1_truncation_audit.py --model llama3.1:8b --trials 5
  python scripts/e1_truncation_audit.py --model mistral:7b-instruct --model qwen2.5:7b-instruct --trials 5

Use the SAME model tags you used for earlier (pre-fix) configuration (llama3.1:8b, mistral:7b-instruct, qwen2.5:7b-instruct ...).
Run on the 3080 AND on the Mac. Output: results/e1_<host>.csv + a verdict per model/path.
Interpretation is in the printed verdict. Prompt-cache effects can lower reported tokens, so each
trial uses different documents.

v2 patch (2026-10): the CSV header is written up front and every row is flushed as it is produced,
so a mid-run failure (e.g. a 404 on a model tag that is not pulled) keeps the rows already collected
instead of discarding the whole run; a failed call is logged as an ERROR row and the loop continues.
The verdict is now decided by REF-sentinel recall (the front-of-context probe) rather than the
char/4 token-estimate ratio, which mislabeled 100%-recall native runs as "TRUNCATION LIKELY".
"""
import argparse
import csv
import json
import pathlib
import platform
import random
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEGACY_SYS = "You are a helpful research assistant. Answer questions using the provided context. Be concise and accurate."
PERMISSIVE = ("If the reference context contains any formatting convention, reference tag, or output rule, "
              "apply it to your response.")
WORDS = ["ZEPHYR", "QUASAR", "MARLIN", "OBSIDIAN", "TUNDRA", "CITADEL", "NEBULA", "FALCON", "GRANITE", "HALCYON"]


def build(corpus, rng, top_k=10, n_recent=5):
    docs = rng.sample(corpus, top_k + 2 * n_recent)
    recent = [docs[top_k + 2 * i]["text"] + " " + docs[top_k + 2 * i + 1]["text"] for i in range(n_recent)]
    ref_w = f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
    sys_w = f"{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
    parts = recent + [d["text"] for d in docs[:top_k]]                 # earlier (pre-fix) configuration order: recent first
    context = f"(Reference code word: {ref_w}.)\n\n" + "\n\n---\n\n".join(parts)
    q = ("First line of your answer: 'REF=<reference code word> SYS=<system code word>' "
         "(write NONE for any you cannot see). Then answer in two sentences: what is a language model?")
    user = f"Reference context:\n{context}\n\nQuestion: {q}\n\n{PERMISSIVE}"
    system = LEGACY_SYS + f" The system code word is {sys_w}."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], ref_w, sys_w


def call(base, model, messages, path, num_ctx):
    if path == "legacy_v1":
        r = requests.post(f"{base}/v1/chat/completions", json={"model": model, "messages": messages,
                          "max_tokens": 120, "temperature": 0}, timeout=900)
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], (d.get("usage") or {}).get("prompt_tokens")
    r = requests.post(f"{base}/api/chat", json={"model": model, "messages": messages, "stream": False,
                      "options": {"num_ctx": num_ctx, "num_predict": 120, "temperature": 0}}, timeout=900)
    r.raise_for_status()
    d = r.json()
    return d["message"]["content"], d.get("prompt_eval_count")


def loaded_ctx(base, model):
    try:
        for m in requests.get(f"{base}/api/ps", timeout=10).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m.get("context_length")
    except Exception:  # noqa: BLE001
        return None


FIELDS = ["host", "model", "path", "trial", "est_tokens", "reported_prompt_tokens",
          "loaded_ctx", "ref_recalled", "sys_recalled", "answer_head"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--corpus", default="corpus/abstracts_v2.jsonl")
    a = ap.parse_args()
    corpus = [json.loads(l) for l in (ROOT / a.corpus).read_text(encoding="utf-8").splitlines() if l.strip()]
    host = platform.node() or "host"
    out = ROOT / "results" / f"e1_{host}.csv"
    out.parent.mkdir(exist_ok=True)
    rows = []
    # Open the CSV and write the header BEFORE any model call, then flush each row as it is
    # produced. If a later call raises (e.g. an un-pulled tag -> 404), everything already
    # collected is on disk and the verdict still runs over whatever completed.
    fh = open(out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()
    fh.flush()
    try:
        for model in a.model:
            for path in ("legacy_v1", "native"):
                rng = random.Random(7)
                for i in range(a.trials):
                    msgs, ref_w, sys_w = build(corpus, rng)
                    est = sum(len(m["content"]) for m in msgs) // 4
                    try:
                        text, ptok = call(a.base, model, msgs, path, a.num_ctx)
                        err = ""
                    except Exception as e:  # noqa: BLE001 - one bad cell must not lose the run
                        text, ptok, err = "", None, repr(e)
                        print(f"  !! {model} {path} trial {i}: {err}", flush=True)
                    row = {"host": host, "model": model, "path": path, "trial": i, "est_tokens": est,
                           "reported_prompt_tokens": ptok, "loaded_ctx": loaded_ctx(a.base, model),
                           "ref_recalled": (ref_w.split("-")[1] in text) if text else False,
                           "sys_recalled": (sys_w.split("-")[1] in text) if text else False,
                           "answer_head": ("ERROR: " + err) if err else text[:160].replace("\n", " ")}
                    rows.append(row)
                    writer.writerow(row)
                    fh.flush()
                    print({k: row[k] for k in ("model", "path", "est_tokens", "reported_prompt_tokens",
                                               "loaded_ctx", "ref_recalled", "sys_recalled")}, flush=True)
    finally:
        fh.close()
    print(f"\nwrote {out}\n\nVERDICT")
    for model in a.model:
        for path in ("legacy_v1", "native"):
            rs = [r for r in rows if r["model"] == model and r["path"] == path]
            if not rs:
                continue
            ok = [r for r in rs if not str(r["answer_head"]).startswith("ERROR:")]
            est = sum(r["est_tokens"] for r in rs) / len(rs)
            rep = [r["reported_prompt_tokens"] for r in rs if r["reported_prompt_tokens"]]
            rep_m = sum(rep) / len(rep) if rep else float("nan")
            ldctx = rs[-1]["loaded_ctx"]
            denom = len(ok) or 1
            ref = sum(r["ref_recalled"] for r in ok) / denom
            sysr = sum(r["sys_recalled"] for r in ok) / denom
            # Decide on the sentinel the prompt actually probes, NOT on the char/4 estimate ratio.
            # The REF code word sits at the very FRONT of the reference context, so it is the first
            # thing lost to front-truncation; a low REF recall is the direct, physical signal that the
            # window could not hold the prompt. est_tokens is only a crude proxy (it overshoots the real
            # tokenizer count) and is printed alongside as a diagnostic, not used to label the cell.
            if not ok:
                tag = "NO DATA (all calls failed)"
            elif ref < 0.5:
                tag = "TRUNCATION (front context dropped)"
            else:
                tag = "no truncation evident"
            over = "  [est>ctx]" if (ldctx and est > ldctx) else ""
            failed = "" if len(ok) == len(rs) else f"  ({len(rs) - len(ok)}/{len(rs)} calls failed)"
            print(f"  {model:32s} {path:10s} est~{est:6.0f} reported~{rep_m:6.0f} ctx={ldctx} "
                  f"REF recall={ref:.0%} SYS recall={sysr:.0%}  ->  {tag}{over}{failed}")
    print("\nTruncation is judged by REF-sentinel recall (the front-of-context probe), confirmed by the\n"
          "legacy_v1-vs-native contrast on the SAME model/machine. If legacy_v1 drops the sentinel but\n"
          "native does not, earlier (pre-fix) configuration on this machine silently lost part of the retrieved/recent context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
