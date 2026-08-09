"""Merge per-task audit TSVs and contrast the gap against sweep behaviour.

The key column is (oracle - mean): how much better the highest-16-step-return
candidate is than an average one, measured in eventual task success.

  positive -> the scoring objective is aligned with task success on this task
  negative -> the objective is misspecified, and ANY search that maximizes it
              hurts, regardless of model quality

If that column's sign matches whether the oracle arm rose or fell in the K
sweep, objective misspecification is established rather than inferred.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Oracle success at K=1 vs its best K, from the completed sweep.
SWEEP_ORACLE = {
    "basketball-v3": (0.62, 0.92), "shelf-place-v3": (0.66, 0.90),
    "stick-pull-v3": (0.50, 0.78), "bin-picking-v3": (0.52, 0.64),
    "peg-insert-side-v3": (0.62, 0.64), "disassemble-v3": (0.78, 0.44),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="experiments/audit")
    a = p.parse_args()

    print(f"{'task':22s} {'n':>4s} {'learned':>8s} {'oracle':>8s} {'mean':>8s} "
          f"{'orc-mean':>9s} {'lrn-mean':>9s} {'sweep dOracle':>14s}")
    gaps, slopes = [], []
    for f in sorted(Path(a.dir).glob("*.tsv")):
        rows = [l.split("\t") for l in f.read_text().splitlines()[1:] if l.strip()]
        if not rows:
            continue
        task = rows[0][0]
        n = int(rows[0][2])
        L = np.mean([float(r[3]) for r in rows])
        O = np.mean([float(r[4]) for r in rows])
        M = np.mean([float(r[5]) for r in rows])
        lo, hi = SWEEP_ORACLE.get(task, (np.nan, np.nan))
        slope = hi - lo
        gaps.append(O - M); slopes.append(slope)
        print(f"{task:22s} {n:4d} {L:8.3f} {O:8.3f} {M:8.3f} "
              f"{O-M:+9.3f} {L-M:+9.3f} {slope:+14.2f}")

    if len(gaps) > 2:
        r = np.corrcoef(gaps, slopes)[0, 1]
        print(f"\ncorr(oracle-mean gap, sweep oracle slope) = {r:+.2f} over "
              f"{len(gaps)} tasks")
        print("a strong positive value means the per-decision audit predicts "
              "which tasks search helps on")


if __name__ == "__main__":
    main()
