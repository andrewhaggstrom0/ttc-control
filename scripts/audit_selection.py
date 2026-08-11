"""Selection regret: what is the learned verifier's pick actually worth?

Both prior correlation metrics were confounded -- the between-episode version by
task difficulty, the within-episode version by position in the episode (return-
to-go shrinks near the end while scores rise near the goal, which makes even the
oracle look anti-correlated).

This avoids proxies entirely. At each decision point we score the SAME candidate
set two ways: with the learned verifier, and with true simulator rollouts. Then
we report, in true-score units:

    picked  = true value of the learned verifier's choice
    best    = true value of the best candidate      (perfect selection)
    mean    = true value of the average candidate   (random selection)

Normalized regret = (best - picked) / (best - mean). At 0 the verifier is
perfect; at 1 it is no better than random; ABOVE 1 it is actively adversarial --
worse than not looking at the scores at all. Watching that cross 1.0 as K grows
is the cleanest statement of the result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from ttc.envs.make import make_env
from ttc.search.base import candidate_seed
from ttc.search.learned import LearnedDynamicsSelector
from ttc.search.oracle import OracleSelector
from scripts.run_sweep import load_dynamics, load_policy, load_value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--k-max", type=int, default=64)
    p.add_argument("--plan-horizon", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out", default=None, help="TSV for plotting")
    a = p.parse_args()

    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    dyn = load_dynamics(f"experiments/ckpt/dyn_{a.task}.pt")
    val = load_value(f"experiments/ckpt/val_{a.task}.pt")
    learned = LearnedDynamicsSelector(dyn, val, beta=a.beta, horizon=a.plan_horizon)
    oracle = OracleSelector()

    ks = [1, 2, 4, 8, 16, 32, a.k_max]
    ks = sorted({k for k in ks if k <= a.k_max})
    stats = defaultdict(lambda: defaultdict(list))

    for ep in range(a.n_episodes):
        env = make_env(a.task, seed=ep)
        obs, _ = env.reset(seed=ep)
        step = 0
        while step < a.max_steps:
            pool = pol.sample(obs, n=a.k_max, seed=candidate_seed(ep, step))

            # True scores for the whole pool, once. Branching is exact, so this
            # does not perturb the trajectory we then follow.
            true_all = oracle.select(obs, pool, env=env,
                                     info={"step": step}).scores
            learn_all = learned.select(obs, pool, info={"step": step}).scores

            for k in ks:
                t, s = true_all[:k], learn_all[:k]
                picked = t[int(np.argmax(s))]
                best, mean = t.max(), t.mean()
                stats[k]["picked"].append(picked)
                stats[k]["best"].append(best)
                stats[k]["mean"].append(mean)
                denom = best - mean
                if abs(denom) > 1e-9:
                    stats[k]["regret"].append((best - picked) / denom)

            # Advance under the learned selector, so the state distribution
            # matches the arm we are auditing.
            chunk = pool[int(np.argmax(learn_all))]
            for act in chunk[:a.n_exec]:
                obs, _, term, trunc, info = env.step(act)
                step += 1
                if term or trunc or info.get("success", 0.0):
                    break
            if term or trunc or info.get("success", 0.0):
                break
        env.close()
        print(f"  episode {ep} done", flush=True)

    print(f"\n{a.task}  beta={a.beta}  h={a.plan_horizon}  "
          f"{len(stats[ks[0]]['picked'])} decisions\n")
    print(f"{'K':>4s} {'picked':>9s} {'best':>9s} {'mean':>9s} "
          f"{'norm regret':>12s}  verdict")
    rows = []
    for k in ks:
        d = stats[k]
        r = float(np.mean(d["regret"])) if d["regret"] else float("nan")
        verdict = ("perfect" if r < 0.15 else
                   "helps" if r < 0.85 else
                   "no better than random" if r <= 1.15 else
                   "ADVERSARIAL")
        print(f"{k:4d} {np.mean(d['picked']):9.3f} {np.mean(d['best']):9.3f} "
              f"{np.mean(d['mean']):9.3f} {r:12.3f}  {verdict}")
        rows.append((k, np.mean(d['picked']), np.mean(d['best']),
                     np.mean(d['mean']), r))

    if a.out:
        import os
        newf = not os.path.exists(a.out)
        with open(a.out, "a") as fh:
            if newf:
                fh.write("task\tk\tpicked\tbest\tmean\tregret\n")
            for k, pk, bs, mn, rg in rows:
                fh.write(f"{a.task}\t{k}\t{pk:.4f}\t{bs:.4f}\t{mn:.4f}\t{rg:.4f}\n")


if __name__ == "__main__":
    main()
