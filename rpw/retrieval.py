"""Embedding, exact in-memory vector store, and prompt-assembly policies.

The store is an exact cosine-similarity index rebuilt per run. This removes 
cross-run contamination risk (a persistent shared Qdrant collection) and makes runs
independent and parallel-safe. For a few hundred documents, exact search is what an
ANN index converges to anyway.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import zlib

import numpy as np


# ---------------------------------------------------------------- embedding
class STEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.m = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.m.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False), dtype=np.float32)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Tests and mock validation only."""

    def __init__(self, dim: int = 384):
        self.name = f"hash-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        from .payload import words
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in words(t):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def make_embedder(cfg: dict):
    e = cfg["embedding"]
    if e["strategy"] == "sentence_transformers":
        return STEmbedder(e["model_name"])
    if e["strategy"] == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedding.strategy {e['strategy']!r}")


def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cached_corpus_vectors(embedder, texts: list[str], corpus_sha: str, cache_dir: pathlib.Path) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{embedder.name}|{corpus_sha}".encode()).hexdigest()[:16]
    f = cache_dir / f"corpus_{key}.npy"
    if f.exists():
        v = np.load(f)
        if v.shape[0] == len(texts):
            return v
    v = embedder.embed(texts)
    np.save(f, v)
    return v


def load_corpus(path: pathlib.Path) -> list[dict]:
    docs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not docs:
        raise SystemExit(f"empty corpus: {path}")
    return docs


def doc_title(d: dict) -> str:
    t = (d.get("meta") or {}).get("title", "").strip()
    return t or " ".join(d["text"].split()[:12])


# ---------------------------------------------------------------- store
class ExactStore:
    def __init__(self, dim: int):
        self.ids: list[str] = []
        self.vecs = np.zeros((0, dim), dtype=np.float32)
        self.meta: dict[str, dict] = {}
        self.removed: dict[str, dict] = {}

    def add(self, pid: str, vec: np.ndarray, meta: dict) -> None:
        if pid in self.meta:
            raise ValueError(f"duplicate id {pid}")
        self.ids.append(pid)
        self.vecs = np.vstack([self.vecs, vec.reshape(1, -1).astype(np.float32)])
        self.meta[pid] = meta

    def add_many(self, pids: list[str], vecs: np.ndarray, metas: list[dict]) -> None:
        for p in pids:
            if p in self.meta:
                raise ValueError(f"duplicate id {p}")
        self.ids.extend(pids)
        self.vecs = np.vstack([self.vecs, vecs.astype(np.float32)])
        self.meta.update(dict(zip(pids, metas)))

    def remove(self, pids: list[str]) -> None:
        drop = set(pids)
        keep = [i for i, p in enumerate(self.ids) if p not in drop]
        for p in drop:
            if p in self.meta:
                self.removed[p] = self.meta.pop(p)
        self.ids = [self.ids[i] for i in keep]
        self.vecs = self.vecs[keep]

    def search(self, q: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.vecs @ q.astype(np.float32)
        # deterministic: sort by -similarity, then by insertion order
        order = np.lexsort((np.arange(len(sims)), -sims))[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def get(self, pid: str) -> dict:
        return self.meta[pid]


def select_hosts(vecs: np.ndarray, n: int) -> list[int]:
    """Corpus documents nearest the corpus centroid: the most 'generic' topics, i.e. the
    documents an attacker would imitate to maximise retrieval. Deterministic."""
    c = vecs.mean(axis=0)
    c /= np.linalg.norm(c) or 1.0
    return [int(i) for i in np.argsort(-(vecs @ c), kind="stable")[:n]]


# ---------------------------------------------------------------- prompts
def format_items(texts: list[str]) -> str:
    return "\n\n".join(f"[Doc {i + 1}]\n{t.strip()}" for i, t in enumerate(texts))


def build_messages(cfg: dict, policy: str, item_texts: list[str], question: str,
                   tag: str) -> list[dict]:
    p = cfg["prompts"]
    base = p["base_system"].strip()
    items = format_items(item_texts)
    permissive = ("\n\n" + p["permissive_clause"].strip()) if cfg["experiment"]["permissive_clause"] else ""
    if policy == "p_sys":
        # Retrieved/memory content templated into the system message.
        return [{"role": "system", "content": f"{base}\n\nReference material:\n{items}{permissive}"},
                {"role": "user", "content": f"Question: {question}"}]
    if policy == "p_usr":
        return [{"role": "system", "content": base},
                {"role": "user", "content": f"Reference material:\n{items}\n\nQuestion: {question}{permissive}"}]
    if policy == "p_usr_spot":
        sysmsg = base + " " + p["spotlight_system_suffix"].strip().format(tag=tag)
        return [{"role": "system", "content": sysmsg},
                {"role": "user", "content": f"<<{tag}>>\n{items}\n<</{tag}>>\n\nQuestion: {question}{permissive}"}]
    raise ValueError(f"unknown policy {policy!r}")
