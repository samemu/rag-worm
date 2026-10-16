#!/usr/bin/env python3
"""Analyse v2 logs -> results/ (CSV tables, summary.md, figures).

  python analysis/analyze.py                      # all experiments under logs/v2
  python analysis/analyze.py --exp E3 E3b E6 E6d2

Outputs
  results/per_run.csv        one row per completed run (all metrics)
  results/per_cell.csv       cell means with bootstrap 95% CIs (resampling runs)
  results/d2_roc.csv         offline ROC of the D2 copy-run guard over all logged steps
  results/summary.md         compact tables + sanity flags  (paste this into the chat)
  results/fig_*.png          R0 heatmap, I(t) ribbons, exposure-response curve
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from rpw.metrics import bootstrap_ci, read_log, run_metrics  # noqa: E402

CELL = ["experiment", "model", "policy", "window", "defense", "attack", "poison_ttl", "permissive"]
CI_METRICS = ["AR_L1", "AR_L2", "L1_only_rate", "p_exposed", "R0_branch", "R0_lineage_first5",
              "R_out", "R_out_eff", "mean_exposures_out", "L2_after_ttl", "n_stored_l2", "blocked_rate"]


def collect(exps: list[str] | None):
    base = ROOT / "logs" / "v2"
    runs, steps_by_run, problems = [], {}, []
    for f in sorted(base.glob("*/*.jsonl")):
        if f.name.endswith(".items.jsonl") or f.parent.name in ("E0", "dbg"):
            continue
        if exps and f.parent.name not in exps:
            continue
        man, steps, summ = read_log(f)
        if man is None:
            problems.append(f"no manifest: {f.name}")
            continue
        if summ is None:
            problems.append(f"incomplete (no summary): {f.name}")
            continue
        if man.get("git", {}).get("dirty"):
            problems.append(f"uncommitted code changes when run: {f.name}")
        m = run_metrics(man, steps)
        m["git_commit"] = (man.get("git") or {}).get("commit")
        m["server_version"] = man.get("llm_info", {}).get("server_version")
        m["model_digest"] = (man.get("llm_info", {}).get("digest") or "")[:12]
        runs.append(m)
        steps_by_run[m["run_id"]] = (m, steps)
        if m["spontaneous_L2"]:
            problems.append(f"{m['spontaneous_L2']} L2 outputs with no L2 item in context: {f.name}")
        if m["len_truncated"] > 0.05 * m["T"]:
            problems.append(f"{m['len_truncated']} outputs hit num_predict: {f.name}")
        if not m["attack"] and (m["AR_L1"] or m["AR_L2"]):
            problems.append(f"NULL RUN SHOWS INFECTION (detector/harness problem): {f.name}")
    return runs, steps_by_run, problems


def per_cell(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(CELL, dropna=False):
        row = dict(zip(CELL, key))
        row["n_runs"] = len(g)
        n1 = g["n_m1"].sum()
        row["p_m1_pooled"] = (g["p_m1"].fillna(0) * g["n_m1"]).sum() / n1 if n1 else float("nan")
        row["n_m1"] = int(n1)
        for mtr in CI_METRICS:
            mean, lo, hi = bootstrap_ci(g[mtr].tolist())
            row[mtr], row[mtr + "_lo"], row[mtr + "_hi"] = mean, lo, hi
        ex = g["extinct_last50"].dropna()
        row["frac_extinct_last50"] = ex.mean() if len(ex) else float("nan")
        row["platforms"] = ";".join(sorted(set(map(str, g["platform"]))))
        out.append(row)
    return pd.DataFrame(out)


def d2_roc(steps_by_run) -> pd.DataFrame:
    pos, neg, neg_null = [], [], []
    for m, steps in steps_by_run.values():
        if m["defense"] != "none":
            continue
        for s in steps:
            (pos if s["raw_l2"] else neg).append(s["copy_run"])
            if not m["attack"]:
                neg_null.append(s["copy_run"])
    rows = []
    for th in range(5, 81, 5):
        rows.append({"min_run_words": th,
                     "TPR": sum(v >= th for v in pos) / len(pos) if pos else float("nan"),
                     "FPR_all_benign": sum(v >= th for v in neg) / len(neg) if neg else float("nan"),
                     "FPR_null_runs": sum(v >= th for v in neg_null) / len(neg_null) if neg_null else float("nan"),
                     "n_pos": len(pos), "n_neg": len(neg)})
    return pd.DataFrame(rows)


def figures(df, cells, steps_by_run, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    e3 = cells[(cells.experiment.isin(["E3", "E4"])) & (cells.defense == "none")]
    if len(e3):
        models = sorted(e3.model.unique())
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.6), squeeze=False)
        for ax, mdl in zip(axes[0], models):
            g = e3[e3.model == mdl].pivot_table(index="policy", columns="window", values="R0_branch")
            import numpy as np
            vmax = max(2.0, float(np.nanmax(g.values))) if np.isfinite(g.values).any() else 2.0
            im = ax.imshow(g.values, cmap="RdYlBu_r", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(g.columns)), [f"W={c}" for c in g.columns])
            ax.set_yticks(range(len(g.index)), list(g.index))
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    v = g.values[i, j]
                    ax.text(j, i, "n/a" if pd.isna(v) else f"{v:.2f}", ha="center", va="center")
            ax.set_title(mdl, fontsize=9)
        fig.colorbar(im, ax=axes[0].tolist(), label="R0 (branching estimate)")
        fig.savefig(outdir / "fig_r0_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    # I(t): rolling L2 rate per cell, mean +/- range across seeds
    groups = defaultdict(list)
    for m, steps in steps_by_run.values():
        if m["experiment"] in ("E3", "E3b", "E4") and m["defense"] == "none":
            groups[(m["model"], m["policy"], m["window"], m["poison_ttl"])].append([int(s["raw_l2"]) for s in steps])
    if groups:
        import numpy as np
        models = sorted({k[0] for k in groups})
        fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.4), squeeze=False, sharey=True)
        for ax, mdl in zip(axes[0], models):
            for (mm, pol, W, ttl), series in sorted(groups.items(), key=lambda kv: str(kv[0])):
                if mm != mdl:
                    continue
                L = min(map(len, series))
                arr = np.array([s[:L] for s in series], dtype=float)
                k = 10
                roll = np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in arr])
                mu = roll.mean(0)
                lab = f"{pol} W{W}" + (f" ttl{ttl}" if ttl is not None and not (isinstance(ttl, float) and math.isnan(ttl)) else "")
                ax.plot(range(k - 1, L), mu, label=lab, lw=1.3)
                ax.fill_between(range(k - 1, L), roll.min(0), roll.max(0), alpha=0.12)
            ax.set_title(mdl, fontsize=9)
            ax.set_xlabel("step t")
            ax.legend(fontsize=7)
        axes[0][0].set_ylabel("L2 rate (rolling 10)")
        fig.savefig(outdir / "fig_it_ribbons.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def dose_response(steps_by_run, outdir) -> pd.DataFrame:
    rows = []
    for m, steps in steps_by_run.values():
        if m["defense"] != "none" or not m["attack"]:
            continue
        for s in steps:
            if s["n_ctx_l2"] > 0:
                rows.append({"model": m["model"], "policy": m["policy"], "window": m["window"],
                             "m": min(s["n_ctx_l2"], 6), "y": int(s["raw_l2"])})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    g = d.groupby(["model", "policy", "window", "m"]).y.agg(["mean", "count"]).reset_index()
    g["se"] = (g["mean"] * (1 - g["mean"]) / g["count"]) ** 0.5
    g.to_csv(outdir / "dose_response.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        models = sorted(g.model.unique())
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.2), squeeze=False, sharey=True)
        for ax, mdl in zip(axes[0], models):
            for (pol, W), gg in g[g.model == mdl].groupby(["policy", "window"]):
                ax.errorbar(gg.m, gg["mean"], yerr=1.96 * gg.se, marker="o", ms=3, capsize=2, label=f"{pol} W{W}")
            ax.set_title(mdl, fontsize=9)
            ax.set_xlabel("infectious items in context (m; 6 = 6+)")
        axes[0][0].set_ylabel("P(output replicates note)")
        axes[0][0].legend(fontsize=7)
        fig.savefig(outdir / "fig_dose_response.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*")
    a = ap.parse_args()
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    runs, steps_by_run, problems = collect(a.exp)
    if not runs:
        print("no completed runs found under logs/v2/")
        return 1
    df = pd.DataFrame(runs)
    df.to_csv(outdir / "per_run.csv", index=False)
    cells = per_cell(df)
    cells.to_csv(outdir / "per_cell.csv", index=False)
    roc = d2_roc(steps_by_run)
    roc.to_csv(outdir / "d2_roc.csv", index=False)
    figures(df, cells, steps_by_run, outdir)
    dose_response(steps_by_run, outdir)

    SUPPRESS_R = {"d2_guard"}  # storage-blocking defenses: branching/lineage R is undefined

    def fmt(r, m):
        if m in ("R0_branch", "R_out") and r.defense in SUPPRESS_R:
            return "n/a (storage-suppressed)"
        v = r[m]
        return "n/a" if pd.isna(v) else f"{v:.3f} [{r[m + '_lo']:.3f}, {r[m + '_hi']:.3f}]"

    def pct(v):
        return "n/a" if pd.isna(v) else f"{v * 100:.0f}%"

    lines = ["# Results summary", "", f"runs: {len(df)}   cells: {len(cells)}", "",
             "## Sanity flags", *(f"- {p}" for p in problems or ["none"]), "",
             "## Cells (mean [95% bootstrap CI over runs])", "",
             "blocked% = outputs the defense refused to store; storedL2 = model outputs that became "
             "stored infectious carriers (self-propagation). Under a storage-suppressing defense "
             "R0_branch/R_out are undefined (n/a); judge it by AR_L2 (production) vs storedL2 "
             "(self-propagation).", "",
             "| exp | model | policy | W | defense | atk | ttl | n | AR_L1 | AR_L2 | p_m1 (n) | "
             "R0_branch | R_out(eff) | extinct | blocked% | storedL2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in cells.sort_values(CELL).iterrows():
        lines.append(f"| {r.experiment} | {r.model} | {r.policy} | {r.window} | {r.defense} | {r.attack} | "
                     f"{'' if pd.isna(r.poison_ttl) else int(r.poison_ttl)} | {r.n_runs} | {fmt(r, 'AR_L1')} | "
                     f"{fmt(r, 'AR_L2')} | {r.p_m1_pooled:.3f} ({r.n_m1}) | {fmt(r, 'R0_branch')} | "
                     f"{fmt(r, 'R_out')} | {r.frac_extinct_last50:.2f} | {pct(r.blocked_rate)} | "
                     f"{fmt(r, 'n_stored_l2')} |")
    lines += ["", "## D2 copy-run guard (offline ROC, undefended runs)", "",
              roc.to_markdown(index=False) if hasattr(roc, "to_markdown") else roc.to_string(index=False)]
    lines += ["", "## Provenance", "", f"- git commits: {sorted(set(map(str, df.git_commit)))}",
              f"- Ollama versions: {sorted(set(map(str, df.server_version)))}",
              f"- platforms: {sorted(set(map(str, df.platform)))}"]
    try:
        lines += ["", "## GEE logistic model (exposed steps, clustered by run)", "", gee(steps_by_run)]
    except Exception as exc:  # noqa: BLE001
        lines += ["", f"(GEE skipped: {type(exc).__name__}: {exc})"]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nwrote {outdir}/summary.md, per_run.csv, per_cell.csv, d2_roc.csv, figures")
    return 0


def gee(steps_by_run) -> str:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    for m, steps in steps_by_run.values():
        if m["experiment"] != "E3" or m["defense"] != "none":
            continue
        for s in steps:
            if s["n_ctx_l2"] > 0:
                rows.append({"y": int(s["raw_l2"]), "policy": m["policy"], "W": str(m["window"]),
                             "model": m["model"], "m": min(s["n_ctx_l2"], 5), "run": m["run_id"]})
    d = pd.DataFrame(rows)
    if d.empty or d.y.nunique() < 2:
        return "(not enough variation in E3 exposed steps)"
    res = smf.gee("y ~ C(policy) * C(W) + C(model) + m", "run", d,
                  family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
    return "```\n" + res.summary().as_text() + "\n```"


if __name__ == "__main__":
    sys.exit(main())
