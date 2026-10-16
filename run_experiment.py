#!/usr/bin/env python3
"""Run ONE experiment.

Examples
  python run_experiment.py --config config/v2/machine_win.yaml --set experiment.policy=p_sys --set experiment.window=5
  python run_experiment.py --set llm.backend=mock --set embedding.strategy=hash --set experiment.T=40
"""
import argparse
import sys

from rpw.config import load_config
from rpw.runner import ContextOverflow, run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="overlay YAML (machine or experiment)")
    ap.add_argument("--set", action="append", default=[], help="key.path=value (repeatable)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    try:
        p = run(cfg, quiet=a.quiet)
    except ContextOverflow as exc:
        print(f"ABORTED (context budget): {exc}", file=sys.stderr)
        return 2
    print(f"log -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
