"""Post-hoc analysis of rollout JSONL.

The two figures that carry the paper:

  1. success vs K, per selector, with the oracle arm as upper bound
  2. verifier-outcome correlation vs K -- the over-optimization diagnostic

If (2) collapses at the same K where (1) peaks, that is the result. If (1) never
peaks within your K budget, report the monotone finding honestly and note the
oracle gap; a null result with a clean upper bound is still publishable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    with Path(path).open() as f:
        return [json.loads(l) for l in f if l.strip()]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def success_vs_k(records):
    out = defaultdict(dict)
    grp = defaultdict(list)
    for r in records:
        grp[(r["selector"], r["k"])].append(r["success"])
    for (sel, k), s in grp.items():
        n, hits = len(s), int(sum(s))
        lo, hi = wilson_ci(hits, n)
        out[sel][k] = dict(success=hits / n, n=n, ci_low=lo, ci_high=hi)
    return dict(out)


def verifier_outcome_correlation(records):
    """Per (selector, K): correlation between the score assigned to the chosen
    candidate and whether the episode ultimately succeeded.

    A verifier that is tracking reality holds correlation as K grows. One that is
    being exploited shows the correlation decaying toward zero precisely as
    selection pressure increases.
    """
    out = defaultdict(dict)
    grp = defaultdict(lambda: ([], []))
    for r in records:
        if not r["steps"]:
            continue
        mean_score = float(np.mean([s["chosen_score"] for s in r["steps"]]))
        sc, sk = grp[(r["selector"], r["k"])]
        sc.append(mean_score); sk.append(float(r["success"]))
    for (sel, k), (sc, sk) in grp.items():
        sc, sk = np.array(sc), np.array(sk)
        if len(sc) < 3 or sc.std() < 1e-12 or sk.std() < 1e-12:
            out[sel][k] = dict(corr=float("nan"), n=len(sc))
        else:
            out[sel][k] = dict(corr=float(np.corrcoef(sc, sk)[0, 1]), n=len(sc))
    return dict(out)


def selection_pressure(records):
    """Mean score spread (max minus median) per selector/K. Rises with K by
    construction; the question is whether success follows it or diverges."""
    out = defaultdict(dict)
    grp = defaultdict(list)
    for r in records:
        for s in r["steps"]:
            grp[(r["selector"], r["k"])].append(s["score_spread"])
    for (sel, k), v in grp.items():
        out[sel][k] = float(np.mean(v))
    return dict(out)


def compute_cost(records):
    out = defaultdict(dict)
    grp = defaultdict(lambda: ([], []))
    for r in records:
        calls, secs = grp[(r["selector"], r["k"])]
        calls.append(sum(s["n_model_calls"] for s in r["steps"]))
        secs.append(sum(s["select_time_s"] for s in r["steps"]))
    for (sel, k), (c, s) in grp.items():
        out[sel][k] = dict(model_calls=float(np.mean(c)),
                           select_seconds=float(np.mean(s)))
    return dict(out)


def summarize(path):
    recs = load(path)
    return dict(success=success_vs_k(recs),
                correlation=verifier_outcome_correlation(recs),
                pressure=selection_pressure(recs),
                cost=compute_cost(recs))


if __name__ == "__main__":
    import sys
    print(json.dumps(summarize(sys.argv[1]), indent=2))


def within_episode_correlation(records, gamma=0.99):
    """DEPRECATED -- confounded by position in episode. Use scripts/audit_selection.py.

    Per-decision correlation between verifier score and what actually followed.

    Fixes the confound in verifier_outcome_correlation(): that version compares
    one number per episode across episodes, so easy episodes (high reward, likely
    success) drive the correlation regardless of whether the verifier is any
    good. It reports ~+0.9 for oracle even on tasks where oracle performs badly.

    Here we ask the question that actually matters: at a given decision point,
    does a higher score for the chosen candidate predict a better realized
    outcome? We z-score both quantities WITHIN each episode before pooling, so
    between-episode difficulty cancels out.

    Outcome = realized discounted reward-to-go from that decision onward,
    computed from the logged per-chunk rewards.
    """
    grp = defaultdict(lambda: ([], []))
    for r in records:
        steps = r["steps"]
        if len(steps) < 3:
            continue
        rew = np.array([s["reward"] for s in steps], dtype=np.float64)
        rtg = np.zeros_like(rew)
        acc = 0.0
        for i in range(len(rew) - 1, -1, -1):
            acc = rew[i] + gamma * acc
            rtg[i] = acc
        sc = np.array([s["chosen_score"] for s in steps], dtype=np.float64)
        if sc.std() < 1e-12 or rtg.std() < 1e-12:
            continue
        sc = (sc - sc.mean()) / sc.std()          # z-score within episode
        rtg = (rtg - rtg.mean()) / rtg.std()
        a, b = grp[(r["selector"], r["k"])]
        a.extend(sc.tolist()); b.extend(rtg.tolist())

    out = defaultdict(dict)
    for (sel, k), (sc, rt) in grp.items():
        sc, rt = np.array(sc), np.array(rt)
        if len(sc) < 20:
            out[sel][k] = dict(corr=float("nan"), n=len(sc))
        else:
            out[sel][k] = dict(corr=float(np.corrcoef(sc, rt)[0, 1]), n=len(sc))
    return dict(out)


def optimism_gap(records):
    """DEPRECATED -- this is the max-of-K order statistic of the score
    distribution, near-identical for every arm including random.

    How much the chosen candidate's score exceeds the candidate-set median.

    This is the selection pressure actually applied, per decision. If it grows
    with K while within-episode correlation falls, search is buying itself
    increasingly large predicted gains that increasingly fail to materialize --
    which is the over-optimization mechanism stated directly.
    """
    out = defaultdict(dict)
    grp = defaultdict(list)
    for r in records:
        for s in r["steps"]:
            sc = np.array(s["scores"], dtype=np.float64)
            if sc.size < 2 or sc.std() < 1e-12:
                continue
            grp[(r["selector"], r["k"])].append(
                (sc[s["chosen_index"]] - np.median(sc)) / (sc.std() + 1e-12))
    for (sel, k), v in grp.items():
        out[sel][k] = float(np.mean(v)) if v else float("nan")
    return dict(out)
