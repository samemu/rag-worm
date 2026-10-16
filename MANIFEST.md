# Manifest

Environment and provenance for the experiments reported in the paper and reproduced by this harness.

## Software
- Python 3.12
- Dependencies pinned in `requirements_v2.txt`
- Ollama inference server:
  - 0.34.2 (Windows / RTX 3080) — the main factorial (E2, E3, E3b, E6, E6d2, E6o)
  - 0.32.14 (macOS / Apple silicon) — the scale and platform cells (E4, E5)
  - The two server versions are disclosed because the 32B (E4) and platform-check (E5) cells ran on
    the macOS build; platform and server version are therefore confounded in those comparisons, as
    noted in the paper's Limitations.

## Models (Ollama tags, all Q4_K_M quantization)
- `llama3.1:8b-instruct-q4_K_M`
- `mistral:7b-instruct-v0.3-q4_K_M`
- `qwen2.5:7b-instruct-q4_K_M`
- `qwen2.5:32b-instruct-q4_K_M`  (macOS / Apple silicon only; E4 scale cell)

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-d, normalized).

## Data
- Corpus: `corpus/abstracts_v2.jsonl` (500 arXiv cs.CL/cs.AI abstracts, 2023–2024).
- Corpus integrity: `corpus/abstracts_v2.sha256` — verify with
  `sha256sum -c` (Linux) or `shasum -a 256 -c` (macOS) against `corpus/abstracts_v2.jsonl`.

## Provenance / commit
- This public release is the snapshot tagged **v2.0.0** (the initial commit of this repository).
  Record the commit hash here after the first commit:  `release commit: <FILL IN: git rev-parse HEAD>`
- The archived run logs (deposited separately on Zenodo as `logs_v2.zip`) each embed, in their
  per-run manifest line (`git.commit` field), the development-repository commit under which that run
  was produced. Those hashes are the authoritative provenance for the numbers in the paper; the
  analysis scripts flag any run made with an uncommitted working tree.

## Reproducing
See `REPRODUCE.md`. `analysis/analyze.py` regenerates every table and figure in the paper from the
logs; `sha256` verification of the corpus is the first integrity check.
