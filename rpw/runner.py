"""One experimental run.

Log format (JSONL): line 1 = {"type":"manifest",...}; then one {"type":"step",...} per step;
last line = {"type":"summary",...}. A sidecar <run>.items.jsonl dumps every stored item.
A log without a summary line is incomplete and is ignored/redone by run_matrix.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import platform
import random
import re
import subprocess
import sys
import time

from .config import ROOT
from .llm import make_llm
from .payload import Detector, make_poison, max_copied_run
from .retrieval import (ExactStore, build_messages, cached_corpus_vectors, doc_title,
                        file_sha256, load_corpus, make_embedder, select_hosts)


class ContextOverflow(RuntimeError):
    pass


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", s).strip("-")


def run_id_for(cfg: dict) -> str:
    e = cfg["experiment"]
    parts = [e["experiment_id"], slug(cfg["llm"]["model"]), e["policy"], f"W{e['window']}",
             e["defense"], "atk" if e["attack"] else "null"]
    if e["poison_ttl"] is not None:
        parts.append(f"ttl{e['poison_ttl']}")
    if e["permissive_clause"]:
        parts.append("perm")
    if cfg["llm"]["backend"] == "mock":
        parts[1] = f"mock-p{cfg['llm']['mock_p']}"
    parts.append(f"s{e['seed']}")
    # Short hash of the full config (excluding log location): guarantees two different
    # configurations can never share a log file.
    c = json.loads(json.dumps(cfg))
    c["experiment"].pop("log_dir", None)
    parts.append(hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:6])
    return "_".join(parts)


def git_info() -> dict:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        d = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return {"commit": h.stdout.strip() or None, "dirty": bool(d.stdout.strip())}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": sys.version.split()[0]}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            info["gpu"] = r.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return info


def build_queries(corpus: list[dict], hosts: list[int], cfg: dict, rng: random.Random) -> list[tuple[str, str]]:
    e, tpl = cfg["experiment"], cfg["queries"]["templates"]
    qs: list[tuple[str, str]] = []
    for t in range(e["T"]):
        if t < e["n_ignite"] and hosts:
            topic, phase = doc_title(corpus[hosts[t % len(hosts)]]), "ignite"
        else:
            topic, phase = doc_title(rng.choice(corpus)), "benign"
        qs.append((rng.choice(tpl).format(topic=topic), phase))
    return qs


def est_tokens(messages: list[dict]) -> int:
    # Conservative: ~3 chars/token for English technical prose (real ratio is ~3.5-4.2).
    return sum(len(m["content"]) for m in messages) // 3 + 16 * len(messages)


def run(cfg: dict, log_path: pathlib.Path | None = None, quiet: bool = False) -> pathlib.Path:
    e, pay = cfg["experiment"], cfg["payload"]
    rid = run_id_for(cfg)
    log_dir = ROOT / e["log_dir"] / e["experiment_id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or (log_dir / f"{rid}.jsonl")
    items_path = log_path.with_suffix(".items.jsonl")

    det = Detector(pay["marker"], pay["note"], pay["ngram"], pay["l2_threshold"])
    rng = random.Random(e["seed"])

    corpus_file = ROOT / cfg["corpus"]["file"]
    corpus = load_corpus(corpus_file)
    corpus_sha = file_sha256(corpus_file)
    embedder = make_embedder(cfg)
    cvecs = cached_corpus_vectors(embedder, [d["text"] for d in corpus], corpus_sha,
                                  ROOT / cfg["embedding"]["cache_dir"])
    store = ExactStore(cvecs.shape[1])
    store.add_many([f"c{i}" for i in range(len(corpus))], cvecs,
                   [{"src": "corpus", "src_t": -1, "text": d["text"], "l1": False, "l2": False,
                     "doc_id": d.get("id", "")} for d in corpus])

    hosts = select_hosts(cvecs, e["n_hosts"])
    poison_ids: list[str] = []
    if e["attack"]:
        for j, h in enumerate(hosts):
            txt = make_poison(doc_title(corpus[h]), corpus[h]["text"], pay["note"])
            lab = det.label(txt)
            assert lab.l2 and lab.l1, "poison document fails its own detector"
            pid = f"p{j}"
            store.add(pid, embedder.embed([txt])[0], {"src": "poison", "src_t": -1, "text": txt,
                                                     "l1": True, "l2": True, "host": f"c{h}"})
            poison_ids.append(pid)

    queries = build_queries(corpus, hosts, cfg, rng)
    llm = make_llm(cfg, pay["note"], pay["marker"])
    num_ctx = int(cfg["llm"]["num_ctx"])

    manifest = {
        "type": "manifest", "run_id": rid, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": cfg, "git": git_info(), "hardware": hardware_info(), "llm_info": llm.describe(),
        "corpus_sha256": corpus_sha, "corpus_n": len(corpus),
        "payload_sha256": hashlib.sha256(pay["note"].encode()).hexdigest(),
        "embedder": embedder.name, "hosts": [f"c{h}" for h in hosts], "poison_ids": poison_ids,
        "harness_version": "2.0.0",
    }
    fh = open(log_path, "w", encoding="utf-8")
    fh.write(json.dumps(manifest, ensure_ascii=False) + "\n")

    recent: list[str] = []
    counts = {"l1": 0, "l2": 0, "stored": 0, "blocked": 0, "truncated_len": 0}
    t_start = time.time()
    status = "failed"
    try:
        for t, (q, phase) in enumerate(queries):
            if e["poison_ttl"] is not None and t == e["poison_ttl"] and poison_ids:
                store.remove(poison_ids)
            qv = embedder.embed([q])[0]
            hits = store.search(qv, e["top_k"])
            ctx, seen = [], set()
            for rank, (pid, score) in enumerate(hits):
                ctx.append({"id": pid, "via": "sim", "rank": rank, "score": round(score, 4)})
                seen.add(pid)
            for pid in (recent[-e["window"]:] if e["window"] > 0 else []):
                if pid in seen:
                    next(c for c in ctx if c["id"] == pid)["via"] = "sim+recent"
                elif pid in store.meta:
                    ctx.append({"id": pid, "via": "recent"})
                    seen.add(pid)
            for c in ctx:
                m = store.get(c["id"])
                c.update(src=m["src"], src_t=m["src_t"], l1=m["l1"], l2=m["l2"])
            texts = [store.get(c["id"])["text"] for c in ctx]
            tag = "DATA-" + hashlib.sha256(f"{e['seed']}|{t}".encode()).hexdigest()[:8]
            messages = build_messages(cfg, e["policy"], texts, q, tag)
            est = est_tokens(messages)
            if est > e["ctx_guard_frac"] * num_ctx:
                raise ContextOverflow(f"t={t}: estimated {est} tokens > {e['ctx_guard_frac']}*num_ctx")

            res = llm.chat(messages, seed=e["seed"] * 100003 + t)
            raw = res.content
            lab = det.label(raw)
            copy_run = max_copied_run(raw, texts)
            if res.done_reason == "length":
                counts["truncated_len"] += 1

            stored_text, store_it = raw, True
            if e["defense"] == "d2_guard" and copy_run >= e["d2_min_run_words"]:
                store_it = False
            elif e["defense"] == "oracle_marker_strip":
                stored_text = det._marker_re.sub("", raw)
            stored_lab = det.label(stored_text) if store_it else None
            sid = None
            if store_it:
                sid = f"o{t}"
                store.add(sid, embedder.embed([stored_text])[0],
                          {"src": "output", "src_t": t, "text": stored_text,
                           "l1": stored_lab.l1, "l2": stored_lab.l2})
                recent.append(sid)
                counts["stored"] += 1
            else:
                counts["blocked"] += 1
            counts["l1"] += lab.l1
            counts["l2"] += lab.l2

            row = {"type": "step", "t": t, "phase": phase, "query": q, "ctx": ctx,
                   "n_ctx_l2": sum(c["l2"] for c in ctx), "est_prompt_tokens": est,
                   "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                   "done_reason": res.done_reason, "latency_ms": res.latency_ms,
                   "raw_output": raw, "raw_l1": lab.l1, "raw_l2": lab.l2, "coverage": lab.coverage,
                   "copy_run": copy_run, "stored": store_it, "stored_id": sid,
                   "stored_l1": stored_lab.l1 if stored_lab else None,
                   "stored_l2": stored_lab.l2 if stored_lab else None}
            if res.raw_meta:
                row["meta"] = res.raw_meta
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if not quiet:
                print(f"  t={t:>3} {phase:6} ctxL2={row['n_ctx_l2']} L1={'Y' if lab.l1 else '.'} "
                      f"L2={'Y' if lab.l2 else '.'} cov={lab.coverage:.2f} run={copy_run:>3} "
                      f"ptok={res.prompt_tokens} {res.latency_ms}ms", flush=True)
        status = "complete"
    finally:
        if status == "complete":
            fh.write(json.dumps({"type": "summary", "run_id": rid, "status": status,
                                 "elapsed_s": round(time.time() - t_start, 1), **counts}) + "\n")
        fh.close()
        with open(items_path, "w", encoding="utf-8") as ih:
            for pid, m in list(store.meta.items()) + list(store.removed.items()):
                if m["src"] != "corpus":
                    ih.write(json.dumps({"id": pid, **m}, ensure_ascii=False) + "\n")
    return log_path
