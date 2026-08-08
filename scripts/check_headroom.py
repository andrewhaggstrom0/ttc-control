"""Measure base-policy success with no search (K=1).

This is THE gate. Target 40-60%. Above ~70% and search has too little room to
show an effect; below ~25% and the policy may be too weak for search to rescue,
which confounds the story differently.

Lever if the number is wrong: demo count (retrain on fewer episodes), not action
noise -- Meta-World's scripted experts are closed-loop and absorb action noise.
"""

from __future__ import annotations

import argparse

import numpy as np

from ttc.envs.make import DEFAULT_TASKS, env_fn
from ttc.eval.rollout import run_episode
from ttc.search.base import FirstSelector
from scripts.run_sweep import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--policy", default="experiments/ckpt/policy.pt")
    p.add_argument("--n-episodes", type=int, default=30)
    a = p.parse_args()

    pol = load_policy(a.policy)
    sel = FirstSelector()
    print(f"{'task':24s} {'success':>8s} {'mean_len':>9s}")
    rates = {}
    for task in a.tasks:
        recs = [run_episode(env_fn(task)(), pol, sel, episode_id=i, k=1,
                            task=task) for i in range(a.n_episodes)]
        r = float(np.mean([x.success for x in recs]))
        L = float(np.mean([x.length for x in recs]))
        rates[task] = r
        print(f"{task:24s} {r:8.2f} {L:9.1f}", flush=True)

    m = float(np.mean(list(rates.values())))
    verdict = ("GOOD - proceed" if 0.30 <= m <= 0.70 else
               "TOO HIGH - retrain on fewer demos" if m > 0.70 else
               "TOO LOW - collect more demos or drop hardest tasks")
    print(f"\nmean base success {m:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
