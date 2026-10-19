# RAG-Worm: a measurement harness for self-replicating instructions in RAG memory

This repository is the experiment harness for the paper *"When Does a Prompt Become a Worm? Measuring
Reproduction Numbers of Self-Replicating Instructions in Retrieval-Augmented Memory."* It measures when
a model-carried instruction that is planted in a retrieval index that also stores the model's own answers,
becomes self-sustaining, and it evaluates defenses against it.

Every log is self-describing (manifest + summary), runs are isolated (a fresh in-memory index per run),
and the context budget not assumed, but rather enforced.

## Layout
    rpw/            config, payload + detectors, retrieval + prompt policies, LLM backends, runner, metrics
    config/v2/      base.yaml (all parameters), plan.yaml (E2–E8), machine_*.yaml, mock.yaml
    run_experiment.py   one run          run_matrix.py   resumable batch for one experiment
    scripts/        e0_validate.py, e1_truncation_audit.py, make_corpus.py
    analysis/       analyze.py (tables, CIs, figures, D2 ROC, GEE), label_sample.py (detector validation)
    tests/          offline unit/integration tests (mock backend)
    corpus/         abstracts_v2.jsonl + .sha256 (built by make_corpus.py)

## Setup
    pip install -r requirements_v2.txt
    python scripts/make_corpus.py          # FIRST; once, on one machine; commit corpus/abstracts_v2.*
    python -m pytest -q tests              # must pass
    python scripts/e0_validate.py          # must print "E0 RESULT: PASS"

Start the Ollama server with a fixed default context (the harness also sends num_ctx per request):
    Windows PowerShell:  $env:OLLAMA_CONTEXT_LENGTH="8192"; ollama serve
    macOS:               OLLAMA_CONTEXT_LENGTH=8192 ollama serve
Pull the exact model tags in config/v2/plan.yaml (same Q4_K_M quantization for every model).

Commit before running (`git commit -am ...`): the manifest records the commit, and analysis flags runs
made with uncommitted changes.

## Run order
    python scripts/e1_truncation_audit.py --model llama3.1:8b --model mistral:7b-instruct --model qwen2.5:7b-instruct
    python run_matrix.py --exp E2  --config config/v2/machine_win.yaml
    python run_matrix.py --exp E3  --config config/v2/machine_win.yaml
    python run_matrix.py --exp E4  --config config/v2/machine_mac.yaml        # Apple silicon
    python run_matrix.py --exp E5  --config config/v2/machine_mac.yaml
    python run_matrix.py --exp E3b --config config/v2/machine_win.yaml
    python run_matrix.py --exp E6  --config config/v2/machine_win.yaml        # also E6d2, E6o
    python analysis/analyze.py

`--dry` lists runs and status. Ctrl-C is safe: rerun the same command to resume. Copy `logs/v2/` from
the Mac into the main repo before analysis.

## Metrics (see rpw/metrics.py for exact definitions)
L1 = marker present (expression). L2 = note reproduced (replication; only L2 can propagate).
p1 = P(L2 | exactly one infectious item in context). R0_branch = p1 × mean exposures per infectious
output. R_out = lineage reproduction number (an R_eff; drifts toward 1 under saturation). AR = attack
rate (high between-seed variance; secondary).

## Reproducing results
`analysis/analyze.py` regenerates `results/` (summary.md, per_cell.csv, per_run.csv, d2_roc.csv, and the
figures) from the logs. See REPRODUCE.md for the archived-logs path and the paper-table mapping.

## Ethics
The payload is a benign formatting instruction with no action, exfiltration, or harmful content. The
work measures and defends against an already-demonstrated class of attack; it introduces no new
offensive technique.

## Citation
See CITATION.cff. If you use this harness, please cite the paper and the archived release.

## Zenodo
10.5281/zenodo.23483303

## License
MIT (code). The corpus is derived from public arXiv abstract metadata; only titles and abstracts are
redistributed.
