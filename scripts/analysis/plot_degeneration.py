"""Replacement for fig4: an uninformative model degenerates to random selection.

The degradation-vs-model-error scatter does not show a usable trend (4 points
per task, heavy noise). What IS clean is the comparison of K-curves: the learned
arm at the worst model quality overlaps the random-selection arm, while better
models degrade well below it. That supports the mechanism -- harm requires a
verifier informative enough to steer -- without claiming a dose-response.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, "scripts/analysis")
from make_figures import series  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="basketball-v3")
    p.add_argument("--degdir", default="experiments/rollouts_degrade_bball")
    p.add_argument("--out", default="experiments/figs/fig4_degeneration.png")
    a = p.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Reference arms from the main sweep.
    s, ks = series(f"experiments/rollouts/{a.task}_rollouts.jsonl")
    for arm, colr, lab, ls in (("first", "#2980b9", "no search", "--"),
                               ("random", "#7f8c8d", "random selection", "-")):
        if arm in s:
            y = [s[arm][k]["success"] for k in ks]
            ax.plot(ks, y, color=colr, ls=ls, lw=2.0, marker="o", ms=4, label=lab)

    shades = {"d0": ("#7b1fa2", "best model"), "d3": ("#e67e22", "worst model")}
    for tag, (colr, lab) in shades.items():
        f = Path(a.degdir) / tag / f"{a.task}_rollouts.jsonl"
        if not f.exists():
            continue
        sd, kd = series(f)
        if "learned" not in sd:
            continue
        y = [sd["learned"][k]["success"] for k in kd]
        ax.plot(kd, y, color=colr, lw=2.4, marker="s", ms=5,
                label=f"learned dynamics, {lab}")

    if "learned" in s:
        y = [s["learned"][k]["success"] for k in ks]
        ax.plot(ks, y, color="#c0392b", lw=2.6, marker="o", ms=5,
                label="learned dynamics, mid model")

    ax.set_xscale("log", base=2)
    ax.set_ylim(0.25, 0.75)
    ax.set_xlabel("K (candidates per decision)")
    ax.set_ylabel("success rate")
    ax.set_title(f"An uninformative model degenerates to random selection\n{a.task}",
                 fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(a.out, dpi=180); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
