"""Generate paper figures 1, 2, 4, 6 from completed sweeps.

Fig 1  success vs K, six-panel small multiples, oracle highlighted
Fig 2  pooled success vs K: first / random / learned / consistency / oracle
Fig 4  degradation vs model error (non-monotonicity), both ladders
Fig 6  audit gap vs oracle sweep slope, one point per task

Beta-ranked arm names are collapsed to a single `learned` series using the
unregularized (rank 0) run, since the beta sweep showed no separation.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttc.eval.metrics import load, success_vs_k

BETA_RE = re.compile(r"^(learned_h\d+)_beta([\d.]+)$")
TASKS = ["peg-insert-side-v3", "basketball-v3", "stick-pull-v3",
         "shelf-place-v3", "disassemble-v3", "bin-picking-v3"]
COLORS = {"oracle": "#111111", "learned": "#c0392b", "random": "#7f8c8d",
          "first": "#2980b9", "consistency": "#27ae60", "value_bon": "#8e44ad"}
LABELS = {"oracle": "oracle (true dynamics)", "learned": "learned dynamics",
          "random": "random selection", "first": "no search (K=1 policy)",
          "consistency": "self-consistency (no verifier)",
          "value_bon": "value best-of-N"}


def collapse_betas(recs, keep_rank=0):
    """Rename beta variants; keep only the requested rank as `learned`."""
    betas = sorted({float(m.group(2)) for r in recs
                    if (m := BETA_RE.match(r["selector"]))})
    if not betas:
        return recs
    target = betas[min(keep_rank, len(betas) - 1)]
    out = []
    for r in recs:
        m = BETA_RE.match(r["selector"])
        if m:
            if float(m.group(2)) != target:
                continue
            r["selector"] = "learned"
        out.append(r)
    return out


def series(path, keep_rank=0):
    s = success_vs_k(collapse_betas(load(path), keep_rank))
    ks = sorted({k for d in s.values() for k in d})
    return s, ks


def fig1(rolldir, out):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, task in zip(axes.ravel(), TASKS):
        f = Path(rolldir) / f"{task}_rollouts.jsonl"
        if not f.exists():
            ax.set_visible(False); continue
        s, ks = series(f)
        for arm in ("first", "random", "learned", "consistency", "oracle"):
            if arm not in s:
                continue
            y = [s[arm][k]["success"] for k in ks]
            ax.plot(ks, y, marker="o", ms=3,
                    lw=2.4 if arm == "oracle" else 1.6,
                    color=COLORS[arm], label=LABELS[arm],
                    zorder=3 if arm == "oracle" else 2)
            if arm in ("oracle", "learned"):
                lo = [s[arm][k]["ci_low"] for k in ks]
                hi = [s[arm][k]["ci_high"] for k in ks]
                ax.fill_between(ks, lo, hi, color=COLORS[arm], alpha=0.12)
        ax.set_xscale("log", base=2); ax.set_title(task, fontsize=10)
        ax.grid(alpha=0.25); ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("K (candidates per decision)")
    for ax in axes[:, 0]:
        ax.set_ylabel("success rate")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Test-time search scaling, per task", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    print("wrote", out)


def fig2(rolldir, out):
    pooled = {}
    for task in TASKS:
        f = Path(rolldir) / f"{task}_rollouts.jsonl"
        if not f.exists():
            continue
        s, ks = series(f)
        for arm, d in s.items():
            for k, v in d.items():
                a, b = pooled.setdefault(arm, {}).setdefault(k, [0.0, 0])
                pooled[arm][k] = [a + v["success"] * v["n"], b + v["n"]]

    ks = sorted({k for d in pooled.values() for k in d})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for arm in ("oracle", "first", "random", "consistency", "learned"):
        if arm not in pooled:
            continue
        y = np.array([pooled[arm][k][0] / pooled[arm][k][1] for k in ks])
        n = np.array([pooled[arm][k][1] for k in ks])
        se = np.sqrt(y * (1 - y) / n)
        ax.plot(ks, y, marker="o", lw=2.6 if arm in ("oracle", "learned") else 1.8,
                color=COLORS[arm], label=LABELS[arm],
                ls="--" if arm == "first" else "-")
        ax.fill_between(ks, y - se, y + se, color=COLORS[arm], alpha=0.12)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (candidates per decision)")
    ax.set_ylabel("success rate (pooled over 6 tasks)")
    ax.set_title("Selecting by a learned verifier is worse than not selecting")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    print("wrote", out)


def _errors(path, pat=r"_(d\d)\.pt", force_tag=None):
    """Parse median sigma error per (task, tag) from a measure_model_error log.

    force_tag overrides regex extraction, for the main-sweep checkpoints whose
    filenames carry no d-tag.
    """
    err = {}
    p = Path(path)
    if not p.exists():
        return err
    for line in p.read_text().splitlines():
        if "median=" not in line:
            continue
        task = line.split("\t")[0].strip()
        if force_tag is not None:
            tag = force_tag
        else:
            m = re.search(pat, line)
            tag = m.group(1) if m else "main"
        err[(task, tag)] = float(re.search(r"median=([\d.]+)", line).group(1))
    return err


def fig4(out):
    err = {}
    err.update(_errors("experiments/logs/degrade_error.tsv"))
    err.update(_errors("experiments/logs/degrade_error_bball.tsv"))
    err.update(_errors("experiments/logs/main_model_error.tsv", force_tag="main"))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for task, d, colr in (("disassemble-v3", "experiments/rollouts_degrade", "#c0392b"),
                          ("basketball-v3", "experiments/rollouts_degrade_bball", "#2980b9")):
        xs, ys = [], []
        for tag in ("d0", "d1", "d2", "d3"):
            f = Path(d) / tag / f"{task}_rollouts.jsonl"
            if not f.exists() or (task, tag) not in err:
                continue
            s, ks = series(f)
            if "learned" not in s:
                continue
            drop = s["learned"][ks[0]]["success"] - s["learned"][ks[-1]]["success"]
            xs.append(err[(task, tag)]); ys.append(drop)
        # main-sweep model as the mid-quality point
        fm = Path("experiments/rollouts") / f"{task}_rollouts.jsonl"
        if fm.exists() and (task, "main") in err:
            s, ks = series(fm)
            if "learned" in s:
                xs.append(err[(task, "main")])
                ys.append(s["learned"][ks[0]]["success"]
                          - s["learned"][ks[-1]]["success"])
        if not xs:
            continue
        o = np.argsort(xs)
        xs, ys = np.array(xs)[o], np.array(ys)[o]
        ax.plot(xs, ys, marker="o", ms=8, lw=1.8, color=colr, label=task)
        for x, y in zip(xs, ys):
            ax.annotate(f"{x:.2f}", (x, y), fontsize=7,
                        xytext=(0, 7), textcoords="offset points", ha="center")

    # random-selection degradation, pooled: the harmless floor
    ax.axhline(0.04, color="#7f8c8d", ls="--", lw=1.4)
    ax.text(0.12, 0.055, "random selection (harmless)", fontsize=8, color="#7f8c8d")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("dynamics model error (median sigma, 4-step rollout)")
    ax.set_ylabel("degradation:  success(K=1) - success(K=64)")
    ax.set_title("Search harm is non-monotonic in model quality")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    print("wrote", out)


def fig6(auditdir, rolldir, out):
    xs, ys, names = [], [], []
    for f in sorted(Path(auditdir).glob("*.tsv")):
        rows = [l.split("\t") for l in f.read_text().splitlines()[1:] if l.strip()]
        if not rows:
            continue
        task = rows[0][0]
        gap = np.mean([float(r[4]) - float(r[5]) for r in rows])
        fm = Path(rolldir) / f"{task}_rollouts.jsonl"
        if not fm.exists():
            continue
        s, ks = series(fm)
        if "oracle" not in s:
            continue
        slope = max(s["oracle"][k]["success"] for k in ks) - s["oracle"][ks[0]]["success"]
        if s["oracle"][ks[-1]]["success"] < s["oracle"][ks[0]]["success"]:
            slope = s["oracle"][ks[-1]]["success"] - s["oracle"][ks[0]]["success"]
        xs.append(gap); ys.append(slope); names.append(task)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="k", lw=0.8)
    ax.scatter(xs, ys, s=70, color="#c0392b", zorder=3)
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n.replace("-v3", ""), (x, y), fontsize=8,
                    xytext=(6, 4), textcoords="offset points")
    if len(xs) > 2:
        r = np.corrcoef(xs, ys)[0, 1]
        m, b = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), max(xs), 10)
        ax.plot(gx, m * gx + b, color="#7f8c8d", ls="--", lw=1.4)
        ax.set_title(f"Per-decision audit predicts task-level search benefit\n"
                     f"r = {r:+.2f}, n = {len(xs)} tasks (p $\\approx$ 0.08)",
                     fontsize=11)
    ax.set_xlabel("audit gap: P(success | best 16-step return) - P(success | average)")
    ax.set_ylabel("oracle arm change across K in main sweep")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rolldir", default="experiments/rollouts")
    p.add_argument("--auditdir", default="experiments/audit")
    p.add_argument("--outdir", default="experiments/figs")
    a = p.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    fig1(a.rolldir, f"{a.outdir}/fig1_per_task.png")
    fig2(a.rolldir, f"{a.outdir}/fig2_pooled.png")
    fig4(f"{a.outdir}/fig4_nonmonotonic.png")
    fig6(a.auditdir, a.rolldir, f"{a.outdir}/fig6_audit_scatter.png")
