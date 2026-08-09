"""Dose-response: does search degrade faster when the dynamics model is worse?

Pairs each degraded model's measured multi-step error (from model_error.tsv)
with how steeply its learned arm falls across K. A monotone relationship turns
"search with a bad model hurts" into a quantitative claim about how bad.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ttc.eval.metrics import load, success_vs_k

BETA_RE = re.compile(r"^(learned_h\d+)_beta([\d.]+)$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="experiments/rollouts_degrade")
    p.add_argument("--errors", default="experiments/logs/degrade_error.tsv")
    a = p.parse_args()

    err = {}
    ep = Path(a.errors)
    if ep.exists():
        for line in ep.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 4 and "median=" in line:
                task = parts[0].strip()
                dyn = parts[1].strip()
                m = re.search(r"_(d\d)\.pt", dyn)
                med = float(re.search(r"median=([\d.]+)", line).group(1))
                if m:
                    err[(task, m.group(1))] = med

    print(f"{'task':22s} {'frac':>6s} {'modelerr':>9s} "
          f"{'K=1':>7s} {'K=8':>7s} {'K=64':>7s} {'drop':>7s}")
    rows = []
    for fdir in sorted(Path(a.dir).glob("d*")):
        frac = fdir.name
        for f in sorted(fdir.glob("*_rollouts.jsonl")):
            recs = load(f)
            for r in recs:
                m = BETA_RE.match(r["selector"])
                if m:
                    r["selector"] = m.group(1)
            s = success_vs_k(recs)
            task = f.name.replace("_rollouts.jsonl", "")
            arm = next((k for k in s if k.startswith("learned")), None)
            if not arm:
                continue
            d = s[arm]
            g = lambda k: d.get(k, {}).get("success", np.nan)
            drop = g(1) - g(64)
            e = err.get((task, frac), np.nan)
            rows.append((e, drop))
            print(f"{task:22s} {frac:>6s} {e:9.3f} "
                  f"{g(1):7.2f} {g(8):7.2f} {g(64):7.2f} {drop:+7.2f}")

    ok = [(e, d) for e, d in rows if np.isfinite(e)]
    if len(ok) > 2:
        e, d = zip(*ok)
        print(f"\ncorr(model error, degradation) = "
              f"{np.corrcoef(e, d)[0,1]:+.2f} over {len(ok)} points")


if __name__ == "__main__":
    main()
