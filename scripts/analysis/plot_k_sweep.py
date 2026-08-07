"""Two-panel figure: success vs K (top), verifier-outcome correlation vs K
(bottom). Aligned x-axes so the reader sees the peak and the collapse line up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttc.eval.metrics import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("--out", default="experiments/figs/k_sweep.png")
    p.add_argument("--title", default="")
    a = p.parse_args()

    s = summarize(a.jsonl)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for sel, d in sorted(s["success"].items()):
        ks = sorted(d)
        y = [d[k]["success"] for k in ks]
        lo = [d[k]["ci_low"] for k in ks]
        hi = [d[k]["ci_high"] for k in ks]
        style = dict(marker="o", lw=2)
        if sel == "oracle":
            style.update(color="black", ls="--", label="oracle (upper bound)")
        else:
            style["label"] = sel
        ax1.plot(ks, y, **style)
        ax1.fill_between(ks, lo, hi, alpha=0.15)

    ax1.set_xscale("log", base=2)
    ax1.set_ylabel("success rate")
    ax1.set_title(a.title or "Test-time compute scaling")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    for sel, d in sorted(s["correlation"].items()):
        ks = sorted(d)
        c = [d[k]["corr"] for k in ks]
        if np.all(np.isnan(c)):
            continue
        ax2.plot(ks, c, marker="s", lw=2, label=sel)
    ax2.axhline(0, color="grey", lw=1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("K (candidates per decision)")
    ax2.set_ylabel("corr(verifier score, outcome)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(a.out, dpi=160)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
