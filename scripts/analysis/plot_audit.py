"""Figure 3: best / picked / mean true-score value vs K.

The widening gap between `best` and `picked` while `picked` tracks `mean` is the
visual statement that better candidates exist and the verifier cannot find them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tsv")
    p.add_argument("--out", default="experiments/figs/fig3_audit.png")
    a = p.parse_args()

    rows = [l.split("\t") for l in Path(a.tsv).read_text().splitlines()[1:] if l.strip()]
    task = rows[0][0]
    ks = [int(r[1]) for r in rows]
    picked = [float(r[2]) for r in rows]
    best = [float(r[3]) for r in rows]
    mean = [float(r[4]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, best, marker="o", lw=2.2, color="#111111", label="best candidate")
    ax.plot(ks, picked, marker="o", lw=2.2, color="#c0392b",
            label="learned verifier's pick")
    ax.plot(ks, mean, marker="o", lw=1.8, color="#7f8c8d", ls="--",
            label="average candidate")
    ax.fill_between(ks, picked, best, color="#c0392b", alpha=0.10)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("K (candidates per decision)")
    ax.set_ylabel("true value (oracle 4-step return)")
    ax.set_title(f"Better candidates appear with K; the verifier finds none\n{task}",
                 fontsize=11)
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=180); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
