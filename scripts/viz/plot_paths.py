"""One panel: the top-down path each candidate leads to.

Reads the npz written by plot_futures.py, so no simulator rollouts are repeated.
Green paths end in success, red in failure, and the violet halo marks the one
the learned verifier chose. Everything else is stripped out.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SUCCESS = "#12795A"
FAIL    = "#B03A22"
PICK    = "#4A3AC4"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="experiments/figs/viz/futures.npz")
    p.add_argument("--out", default="experiments/figs/viz/paths.png")
    p.add_argument("--who", default="learned",
                   choices=["learned", "oracle", "random"])
    a = p.parse_args()

    d = np.load(a.data, allow_pickle=True)
    paths = [np.asarray(x, dtype=float) for x in d["paths"]]
    succ = d["success"]
    picks = dict(zip(("learned", "oracle", "random"), d["picks"]))
    pick = int(picks[a.who])
    K = len(paths)

    fig, ax = plt.subplots(figsize=(8.4, 7))

    # Failures first so successful futures and the highlighted pick sit on top.
    for i in sorted(range(K), key=lambda j: (j == pick, succ[j])):
        path, is_pick = paths[i], i == pick
        color = SUCCESS if succ[i] else FAIL
        if is_pick:
            ax.plot(path[:, 0], path[:, 1], color=PICK, lw=7, alpha=.30,
                    solid_capstyle="round", zorder=4)
        ax.plot(path[:, 0], path[:, 1], color=color,
                lw=2.6 if is_pick else 1.2, alpha=1.0 if is_pick else .55,
                solid_capstyle="round", zorder=5 if is_pick else 2)
        ax.plot(path[-1, 0], path[-1, 1], "o", color=color,
                ms=9 if is_pick else 4, mec="white", mew=.9,
                zorder=6 if is_pick else 3)

    start = paths[0][0]
    ax.plot(start[0], start[1], "o", color="#141A21", ms=11, zorder=7)
    ax.annotate("the robot chooses here",
                (start[0], start[1]), xytext=(20, 20),
                textcoords="offset points", fontsize=10.5,
                arrowprops=dict(arrowstyle="-", lw=.9, color="#5A6672"))

    n_ok = int(succ.sum())
    ax.set_title(
        f"Where each of {K} choices leads\n"
        f"{n_ok} succeed, {K - n_ok} fail \u2014 the {a.who} verifier chose one that "
        f"{'succeeds' if succ[pick] else 'fails'}",
        fontsize=13, loc="left", pad=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.18)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(handles=[
        Line2D([], [], color=SUCCESS, lw=2.6, label="leads to success"),
        Line2D([], [], color=FAIL, lw=2.6, label="leads to failure"),
        Line2D([], [], color=PICK, lw=7, alpha=.40, label=f"chosen ({a.who})"),
    ], loc="best", frameon=False, fontsize=10)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}  ({n_ok}/{K} succeed, {a.who} picked #{pick})")


if __name__ == "__main__":
    main()
