"""Cross-task summary: success vs K, and where each arm peaks.

Prints one table per task plus a pooled view. The columns that matter:
  - oracle should climb monotonically (perfect model, more search = better)
  - learned_* arms peaking at some K < 64 is the over-optimization result
  - corr is verifier-score vs actual outcome; collapse at the peak K is the
    mechanism, not just the symptom
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ttc.eval.metrics import load, success_vs_k, verifier_outcome_correlation


def table(name, succ, corr):
    ks = sorted({k for d in succ.values() for k in d})
    print(f"\n=== {name}")
    print("arm".ljust(24) + "".join(f"K={k}".rjust(8) for k in ks) + "   peak")
    for arm in sorted(succ):
        row, vals = "", []
        for k in ks:
            v = succ[arm].get(k, {}).get("success")
            vals.append(v if v is not None else np.nan)
            row += ("  --  " if v is None else f"{v:.2f}").rjust(8)
        best = ks[int(np.nanargmax(vals))]
        turned = best < max(ks) and vals[-1] < np.nanmax(vals) - 0.05
        print(arm.ljust(24) + row + f"   K={best}" + ("  TURNOVER" if turned else ""))
    print("\ncorrelation(verifier score, outcome):")
    for arm in sorted(corr):
        cs = [corr[arm].get(k, {}).get("corr", np.nan) for k in ks]
        if np.all(np.isnan(cs)):
            continue
        print(arm.ljust(24) + "".join(
            ("  nan " if np.isnan(c) else f"{c:+.2f}").rjust(8) for c in cs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="experiments/rollouts")
    a = p.parse_args()

    pooled = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for f in sorted(Path(a.dir).glob("*_rollouts.jsonl")):
        recs = load(f)
        s, c = success_vs_k(recs), verifier_outcome_correlation(recs)
        table(f.name.replace("_rollouts.jsonl", ""), s, c)
        for arm, d in s.items():
            for k, v in d.items():
                pooled[arm][k][0] += v["success"] * v["n"]
                pooled[arm][k][1] += v["n"]

    ks = sorted({k for d in pooled.values() for k in d})
    print("\n\n=== POOLED ACROSS TASKS")
    print("arm".ljust(24) + "".join(f"K={k}".rjust(8) for k in ks) + "   peak")
    for arm in sorted(pooled):
        vals = [pooled[arm][k][0] / pooled[arm][k][1] if pooled[arm][k][1] else np.nan
                for k in ks]
        best = ks[int(np.nanargmax(vals))]
        turned = best < max(ks) and vals[-1] < np.nanmax(vals) - 0.05
        print(arm.ljust(24) + "".join(f"{v:.2f}".rjust(8) for v in vals)
              + f"   K={best}" + ("  TURNOVER" if turned else ""))


if __name__ == "__main__":
    main()
