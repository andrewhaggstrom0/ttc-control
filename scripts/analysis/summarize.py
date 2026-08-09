"""Cross-task summary with beta ranking and un-confounded correlation.

Beta values were calibrated per task, so the raw selector names differ across
tasks and cannot be pooled. We rank each task's betas ascending and rename them
beta_rank0..3, where rank0 is always the unregularized (beta=0) control.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ttc.eval.metrics import (load, optimism_gap, success_vs_k,
                              within_episode_correlation)

BETA_RE = re.compile(r"^(learned_h\d+)_beta([\d.]+)$")


def rank_betas(records):
    """Rewrite per-task beta names to beta_rank0..3 so tasks are poolable."""
    betas = sorted({float(m.group(2)) for r in records
                    if (m := BETA_RE.match(r["selector"]))})
    rank = {b: i for i, b in enumerate(betas)}
    for r in records:
        m = BETA_RE.match(r["selector"])
        if m:
            r["selector"] = f"{m.group(1)}_rank{rank[float(m.group(2))]}"
    return records, betas


def grid(title, d, ks, fmt="{:.2f}", flag=True):
    print(f"\n{title}")
    for arm in sorted(d):
        vals = [d[arm].get(k) for k in ks]
        vals = [v["success"] if isinstance(v, dict) and "success" in v
                else v["corr"] if isinstance(v, dict) else v for v in vals]
        row = "".join(("  --  " if v is None or (isinstance(v, float) and np.isnan(v))
                       else fmt.format(v)).rjust(8) for v in vals)
        tag = ""
        if flag:
            arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
            if not np.all(np.isnan(arr)):
                best = ks[int(np.nanargmax(arr))]
                tag = f"   peak K={best}"
                if best < max(ks) and arr[-1] < np.nanmax(arr) - 0.05:
                    tag += "  TURNOVER"
        print(arm.ljust(26) + row + tag)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="experiments/rollouts")
    a = p.parse_args()

    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    pooled_corr = defaultdict(lambda: defaultdict(list))
    ks_all = set()

    for f in sorted(Path(a.dir).glob("*_rollouts.jsonl")):
        recs, betas = rank_betas(load(f))
        name = f.name.replace("_rollouts.jsonl", "")
        s = success_vs_k(recs)
        c = within_episode_correlation(recs)
        g = optimism_gap(recs)
        ks = sorted({k for d in s.values() for k in d})
        ks_all |= set(ks)

        print(f"\n{'='*70}\n=== {name}   betas={[round(b,2) for b in betas]}")
        print("arm".ljust(26) + "".join(f"K={k}".rjust(8) for k in ks))
        grid("SUCCESS", s, ks)
        grid("WITHIN-EPISODE CORR (verifier score vs realized return-to-go)",
             c, ks, "{:+.2f}", flag=False)
        grid("OPTIMISM GAP (chosen score above median, in sd)",
             g, ks, "{:.2f}", flag=False)

        for arm, d in s.items():
            for k, v in d.items():
                pooled[arm][k][0] += v["success"] * v["n"]
                pooled[arm][k][1] += v["n"]
        for arm, d in c.items():
            for k, v in d.items():
                if not np.isnan(v["corr"]):
                    pooled_corr[arm][k].append(v["corr"])

    ks = sorted(ks_all)
    print(f"\n{'='*70}\n=== POOLED ACROSS TASKS")
    print("arm".ljust(26) + "".join(f"K={k}".rjust(8) for k in ks))
    grid("SUCCESS", {a_: {k: {"success": v[0] / v[1]} for k, v in d.items() if v[1]}
                     for a_, d in pooled.items()}, ks)
    grid("WITHIN-EPISODE CORR", {a_: {k: {"corr": float(np.mean(v))}
                                      for k, v in d.items()}
                                 for a_, d in pooled_corr.items()},
         ks, "{:+.2f}", flag=False)


if __name__ == "__main__":
    main()
