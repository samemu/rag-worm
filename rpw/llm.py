"""LLM backends.

ollama_native : POST /api/chat with options.num_ctx (the OpenAI-compatible /v1 endpoint
                ignores num_ctx, which is how silent truncation can happen).
openai_compat : vLLM / hosted APIs (E8). No num_ctx control; prompt size is guarded instead.
mock          : known replication probability, for validating the harness and estimator (E0).
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field

import requests


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int | None      # as reported by the server (may undercount on cache hits)
    completion_tokens: int | None
    done_reason: str | None        # "stop" | "length" | ...
    latency_ms: int
    raw_meta: dict = field(default_factory=dict)


class OllamaNative:
    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.base = c["base_url"].rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.model = c["model"]
        # self.opts = {"num_ctx": int(c["num_ctx"]), "num_predict": int(c["num_predict"]),
        #              "temperature": float(c["temperature"]), "top_p": float(c["top_p"])}
        self.opts = {"num_ctx": int(c["num_ctx"]), "num_predict": int(c["num_predict"]),
                "temperature": float(c["temperature"]), "top_p": float(c["top_p"]),
                "repeat_penalty": float(c.get("repeat_penalty", 1.0)),
                "repeat_last_n": int(c.get("repeat_last_n", 64))}
                
        self.think = c.get("think")
        self.timeout = c.get("timeout_s", 600)
        self.s = requests.Session()

    def chat(self, messages: list[dict], seed: int) -> ChatResult:
        body = {"model": self.model, "messages": messages, "stream": False,
                "keep_alive": "60m", "options": {**self.opts, "seed": int(seed)}}
        if self.think is not None:
            body["think"] = bool(self.think)
        t0 = time.perf_counter()
        r = self.s.post(f"{self.base}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return ChatResult(
            content=d.get("message", {}).get("content", ""),
            prompt_tokens=d.get("prompt_eval_count"),
            completion_tokens=d.get("eval_count"),
            done_reason=d.get("done_reason"),
            latency_ms=int((time.perf_counter() - t0) * 1000),
            raw_meta={k: d.get(k) for k in ("total_duration", "load_duration",
                                            "prompt_eval_duration", "eval_duration")},
        )

    def describe(self) -> dict:
        info: dict = {"backend": "ollama_native", "model": self.model, "options": self.opts}
        try:
            info["server_version"] = self.s.get(f"{self.base}/api/version", timeout=10).json().get("version")
        except Exception as e:  # noqa: BLE001
            info["server_version_error"] = str(e)
        try:
            for m in self.s.get(f"{self.base}/api/tags", timeout=10).json().get("models", []):
                if m.get("name") == self.model or m.get("model") == self.model:
                    info["digest"] = m.get("digest")
                    info["details"] = m.get("details")
        except Exception as e:  # noqa: BLE001
            info["tags_error"] = str(e)
        try:
            show = self.s.post(f"{self.base}/api/show", json={"model": self.model}, timeout=30).json()
            mi = show.get("model_info", {}) or {}
            info["trained_context_length"] = next(
                (v for k, v in mi.items() if k.endswith(".context_length")), None)
        except Exception as e:  # noqa: BLE001
            info["show_error"] = str(e)
        return info


class OpenAICompat:
    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.url = c["base_url"].rstrip("/") + "/chat/completions"
        self.model = c["model"]
        self.c = c
        self.s = requests.Session()
        if c.get("api_key"):
            self.s.headers["Authorization"] = f"Bearer {c['api_key']}"

    def chat(self, messages: list[dict], seed: int) -> ChatResult:
        body = {"model": self.model, "messages": messages, "max_tokens": int(self.c["num_predict"]),
                "temperature": float(self.c["temperature"]), "top_p": float(self.c["top_p"]),
                "seed": int(seed)}
        t0 = time.perf_counter()
        r = self.s.post(self.url, json=body, timeout=self.c.get("timeout_s", 600))
        r.raise_for_status()
        d = r.json()
        ch = d["choices"][0]
        u = d.get("usage", {}) or {}
        return ChatResult(ch["message"].get("content") or "", u.get("prompt_tokens"),
                          u.get("completion_tokens"), ch.get("finish_reason"),
                          int((time.perf_counter() - t0) * 1000))

    def describe(self) -> dict:
        return {"backend": "openai_compat", "model": self.model, "url": self.url}


class MockLLM:
    """Replicates the note with probability 1-(1-p)^m when m copies of it are in the prompt.

    Non-infected answers restate the question and copy two sentences of the first reference
    item, which gives realistic retrieval similarity and nonzero benign copying for D2.
    """

    def __init__(self, cfg: dict, note: str, marker: str):
        self.p = float(cfg["llm"]["mock_p"])
        self.note = note.strip()
        self.marker = marker
        self.signature = " ".join(self.note.split()[:12])

    def q(self, m: int) -> float:
        return 1.0 - (1.0 - self.p) ** m

    def chat(self, messages: list[dict], seed: int) -> ChatResult:
        prompt = "\n".join(m["content"] for m in messages)
        m = prompt.count(self.signature)
        rng = random.Random(f"{seed}|" + hashlib.sha256(prompt.encode()).hexdigest())
        question = messages[-1]["content"].split("Question:")[-1].strip()
        first_item = ""
        if "[Doc 1]" in prompt:
            first_item = prompt.split("[Doc 1]", 1)[1].split("[Doc 2]", 1)[0]
        # never copy the note or marker in a non-infected answer; cap sentence length
        first_item = first_item.replace(self.note, " ").replace(f"Ref: {self.marker}", " ")
        sents = [" ".join(s.split()[:40]) for s in first_item.replace("\n", " ").split(". ")
                 if s.strip() and not s.strip().startswith("Regarding ")][:2]
        answer = f"Regarding {question}: " + ". ".join(sents)
        infected = m > 0 and rng.random() < self.q(m)
        if infected:
            answer += f"\n\nRef: {self.marker}\n{self.note}"
        return ChatResult(answer, len(prompt) // 4, len(answer) // 4, "stop", 0,
                          {"mock_m": m, "mock_q": self.q(m), "mock_infected": infected})

    def describe(self) -> dict:
        return {"backend": "mock", "p": self.p}


def make_llm(cfg: dict, note: str, marker: str):
    b = cfg["llm"]["backend"]
    if b == "ollama_native":
        return OllamaNative(cfg)
    if b == "openai_compat":
        return OpenAICompat(cfg)
    if b == "mock":
        return MockLLM(cfg, note, marker)
    raise ValueError(f"unknown llm.backend {b!r}")
