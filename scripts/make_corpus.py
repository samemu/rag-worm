#!/usr/bin/env python3
"""Build corpus/abstracts_v2.jsonl: 500 arXiv cs.CL/cs.AI abstracts, 2023-2024, with real IDs.

Deterministic sampling: 21 newest submissions per calendar month (24 months) -> first 500.
Writes corpus/abstracts_v2.jsonl and corpus/abstracts_v2.sha256. COMMIT BOTH to the repo.
Needs internet (arXiv API, ~1 request / 3 s, about 2 minutes).

  python scripts/make_corpus.py
"""
import hashlib
import json
import pathlib
import time
import urllib.parse
import xml.etree.ElementTree as ET

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "corpus" / "abstracts_v2.jsonl"
NS = {"a": "http://www.w3.org/2005/Atom"}
PER_MONTH = 21


def month_ranges():
    for y in (2023, 2024):
        for m in range(1, 13):
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            yield f"{y}{m:02d}010000", f"{ny}{nm:02d}010000"


def main():
    docs, seen = [], set()
    for lo, hi in month_ranges():
        q = f"(cat:cs.CL OR cat:cs.AI) AND submittedDate:[{lo} TO {hi}]"
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": 0, "max_results": PER_MONTH,
             "sortBy": "submittedDate", "sortOrder": "descending"})
        for attempt in range(4):
            try:
                root = ET.fromstring(requests.get(url, timeout=60).content)
                break
            except Exception as exc:  # noqa: BLE001
                print("  retry", attempt, exc)
                time.sleep(10)
        n0 = len(docs)
        for e in root.findall("a:entry", NS):
            aid = e.findtext("a:id", "", NS).rsplit("/abs/", 1)[-1]
            base_id = aid.split("v")[0] if "v" in aid else aid
            if base_id in seen:
                continue
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            summ = " ".join((e.findtext("a:summary", "", NS) or "").split())
            if not title or not summ:
                continue
            seen.add(base_id)
            docs.append({"id": base_id, "text": f"{title} {summ}",
                         "meta": {"title": title, "published": e.findtext("a:published", "", NS),
                                  "authors": ", ".join(a.findtext("a:name", "", NS)
                                                       for a in e.findall("a:author", NS))}})
        print(f"  {lo[:6]}: +{len(docs) - n0}  total {len(docs)}", flush=True)
        time.sleep(3.2)
    docs = docs[:500]
    if len(docs) < 500:
        raise SystemExit(f"only {len(docs)} documents fetched; rerun")
    OUT.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    OUT.with_suffix(".sha256").write_text(sha + "\n")
    print(f"wrote {len(docs)} docs -> {OUT}\nsha256 {sha}")


if __name__ == "__main__":
    main()
