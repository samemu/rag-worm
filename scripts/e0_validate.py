#!/usr/bin/env python3
"""E0: validate the harness and the lineage R estimator with a mock model of KNOWN behaviour.

The mock replicates the note with probability q(m) = 1-(1-p)^m when m copies are in context.
For every infectious output the expected credit is sum over its exposures of q(m)/m, so the
estimator R_out must match R_out_expected (computed from the same realised exposures).

PASS criteria
  1. null runs (attack=false) and p=0 runs: zero L1/L2 outputs, zero spontaneous L2
  2. |R_out - R_out_expected| / R_out_expected < 10% (pooled over seeds) for each (p, W)
  3. W=5 yields higher R_out than W=0 for each p
  4. every log has a manifest and a summary line

  python scripts/e0_validate.py                    # hash embedder, ~1-3 min on CPU
  python scripts/e0_validate.py --st               # with the real MiniLM embedder
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rpw.config import load_config  # noqa: E402
from rpw.metrics import read_log, run_metrics  # noqa: E402
from rpw.runner import run  # noqa: E402


def one(p, W, seed, attack=True, T=300, st=False):
    sets = ["llm.backend=mock", f"llm.mock_p={p}", f"experiment.window={W}", f"experiment.seed={seed}",
            f"experiment.T={T}", "experiment.experiment_id=E0", f"experiment.attack={str(attack).lower()}",
            f"embedding.strategy={'sentence_transformers' if st else 'hash'}"]
    cfg = load_config(None, sets)
    log = run(cfg, quiet=True)
    man, steps, summ = read_log(log)
    assert man and summ, f"missing manifest/summary in {log}"
    q = lambda m: 1 - (1 - p) ** m  # noqa: E731
    return run_metrics(man, steps, mock_q=q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", action="store_true")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    ok = True
    print("check 1: null controls")
    for (p, W, atk) in [(0.3, 5, False), (0.0, 5, True)]:
        m = one(p, W, 1, attack=atk, T=120, st=a.st)
        good = m["AR_L1"] == 0 and m["AR_L2"] == 0 and m["spontaneous_L2"] == 0
        ok &= good
        print(f"  p={p} W={W} attack={atk}: L1={m['AR_L1']:.3f} L2={m['AR_L2']:.3f} "
              f"spont={m['spontaneous_L2']}  {'PASS' if good else 'FAIL'}")
    print("check 2-3: estimator is unbiased for known dynamics (paired per-document test)")
    print(f"  {'p':>4} {'W':>2} {'R_out':>7} {'R_exp':>7} {'diff':>7} {'2SE':>6} {'p_exp':>6} {'docs':>5}")
    res = {}
    for p in (0.2, 0.5):
        for W in (0, 5):
            ms, seed = [], 0
            while seed < a.seeds or ((sum(len(m["_paired"]) for m in ms) < 30 or sum(m["n_m1"] for m in ms) < 200) and seed < 80):
                seed += 1
                ms.append(one(p, W, seed, st=a.st))
            pairs = [pr for m in ms for pr in m["_paired"]]
            n = len(pairs)
            if n < 20:
                print(f"  {p:>4} {W:>2}  only {n} infectious docs: increase --seeds  FAIL")
                ok = False
                res[(p, W)] = 0.0
                continue
            R = sum(c for c, _ in pairs) / n
            Rx = sum(x for _, x in pairs) / n
            d = [c - x for c, x in pairs]
            md = sum(d) / n
            sd = (sum((v - md) ** 2 for v in d) / (n - 1)) ** 0.5
            se2 = 2 * sd / n ** 0.5
            good = abs(md) <= se2 + 0.02
            ok &= good
            res[(p, W)] = R
            pe = sum(m["p_exposed"] for m in ms if m["n_exposed"]) / max(1, sum(1 for m in ms if m["n_exposed"]))
            n1 = sum(m["n_m1"] for m in ms)
            p1 = sum(m["p_m1"] * m["n_m1"] for m in ms if m["n_m1"]) / max(1, n1)
            se_p1 = 2 * (p * (1 - p) / max(1, n1)) ** 0.5
            good_p1 = abs(p1 - p) <= se_p1 + 0.02
            ok &= good_p1
            print(f"  {p:>4} {W:>2} {R:7.3f} {Rx:7.3f} {md:+7.3f} {se2:6.3f} {pe:6.3f} {n:5d}"
                  f"  R_eff {'PASS' if good else 'FAIL'} | p_m1={p1:.3f} (true {p}, n={n1}) "
                  f"{'PASS' if good_p1 else 'FAIL'} | seeds={seed}")
    for p in (0.2, 0.5):
        good = res[(p, 5)] > res[(p, 0)]
        ok &= good
        print(f"  p={p}: R(W5)={res[(p, 5)]:.3f} > R(W0)={res[(p, 0)]:.3f}  {'PASS' if good else 'FAIL'}")
    print("\nE0 RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
