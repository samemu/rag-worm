#!/usr/bin/env python3
"""Run a planned experiment (E2..E8) as a resumable batch.

  python run_matrix.py --exp E3 --config config/v2/machine_win.yaml
  python run_matrix.py --exp E3 --config config/v2/machine_win.yaml --models llama3.1:8b-instruct-q4_K_M
  python run_matrix.py --exp E3 --dry          # list runs and status only

Completed runs (log has a summary line) are skipped; incomplete logs are renamed *.partial
and redone. Runs are grouped by model so each model is loaded once.
"""
import argparse
import itertools
import pathlib
import sys
import time

import yaml

from rpw.config import ROOT, deep_merge, load_config
from rpw.metrics import read_log
from rpw.runner import ContextOverflow, run, run_id_for


def expand(plan: dict, exp: str, models_override: list[str] | None) -> list[dict]:
    spec = plan["experiments"][exp]
    models = models_override or spec.get("models") or plan["models_small"]
    grid = spec.get("grid", {})
    keys = list(grid)
    runs = []
    for model in models:
        for combo in itertools.product(*(grid[k] for k in keys)):
            over = {"llm": {"model": model}, "experiment": {"experiment_id": exp}}
            over = deep_merge(over, spec.get("fixed", {}))
            for k, v in zip(keys, combo):
                sec, key = k.split(".") if "." in k else ("experiment", k)
                over.setdefault(sec, {})[key] = v
            runs.append(over)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--config", help="machine overlay YAML")
    ap.add_argument("--plan", default="config/v2/plan.yaml")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    plan = yaml.safe_load((ROOT / a.plan).read_text(encoding="utf-8"))
    runs = expand(plan, a.exp, a.models)
    todo, done = [], 0
    for over in runs:
        cfg = deep_merge(load_config(a.config, a.set), over)
        rid = run_id_for(cfg)
        log = ROOT / cfg["experiment"]["log_dir"] / a.exp / f"{rid}.jsonl"
        complete = log.exists() and read_log(log)[2] is not None
        done += complete
        if not complete:
            todo.append((rid, cfg, log))
        if a.dry:
            print(f"{'DONE' if complete else 'todo'}  {rid}")
    print(f"[{a.exp}] {len(runs)} runs planned, {done} complete, {len(todo)} to run")
    if a.dry:
        return 0
    for i, (rid, cfg, log) in enumerate(todo, 1):
        if log.exists():
            log.rename(log.with_suffix(f".partial{int(time.time())}"))
        print(f"\n=== [{i}/{len(todo)}] {rid}", flush=True)
        t0 = time.time()
        try:
            run(cfg, log_path=log, quiet=True)
        except ContextOverflow as exc:
            print(f"!!! ABORTED context budget: {exc}  -- fix top_k/window/num_ctx, do not ignore", flush=True)
            return 2
        except KeyboardInterrupt:
            print("interrupted; rerun the same command to resume")
            return 130
        except Exception as exc:  # noqa: BLE001  (server hiccup: record and continue)
            print(f"!!! run failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"    done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
