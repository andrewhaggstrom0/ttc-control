"""Pick beta values from measured scales rather than guesses.

The learned selector scores candidates as (return - beta * disagreement). For
the beta sweep to be informative, beta must be scaled so the penalty is
comparable to the return spread across candidates at a typical decision point.
We roll the policy out, record both spreads, and report betas that make the
penalty worth ~0.25x, 1x, and 4x the return spread.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ttc.envs.make import make_env
from scripts.run_sweep import load_dynamics, load_policy, load_value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="stick-pull-v3")
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--dynamics", default=None,
                   help="checkpoint to calibrate against; defaults to the "
                        "task's full-data model. Degraded models have a "
                        "different disagreement scale, so calibrating "
                        "against the wrong one confounds model quality "
                        "with regularization strength.")
    a = p.parse_args()

    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    dyn = load_dynamics(a.dynamics or f"experiments/ckpt/dyn_{a.task}.pt")
    val = load_value(f"experiments/ckpt/val_{a.task}.pt")

    ret_spreads, dis_spreads, dis_all = [], [], []
    for ep in range(a.n_episodes):
        env = make_env(a.task, seed=ep)
        obs, _ = env.reset(seed=ep)
        for step in range(0, 300, 8):
            c = pol.sample(obs, n=a.k, seed=ep * 1000 + step)
            o = torch.as_tensor(obs, device="cuda").float().reshape(-1)
            ct = torch.as_tensor(c, device="cuda").float()
            ret, dis, _ = dyn.rollout(o, ct[:, :a.horizon])
            ret_spreads.append((ret.max() - ret.min()).item())
            dis_spreads.append((dis.max() - dis.min()).item())
            dis_all.append(dis.mean().item())
            for act in c[0][:8]:
                obs, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
            if term or trunc:
                break
        env.close()

    R, D, M = map(np.array, (ret_spreads, dis_spreads, dis_all))
    print(f"task {a.task}, K={a.k}, {len(R)} decision points\n")
    print(f"return spread across candidates  median={np.median(R):.4f} "
          f"p90={np.percentile(R, 90):.4f}")
    print(f"disagreement spread              median={np.median(D):.4f} "
          f"p90={np.percentile(D, 90):.4f}")
    print(f"disagreement level               median={np.median(M):.4f} "
          f"max={M.max():.4f}")

    base = np.median(R) / max(np.median(D), 1e-9)
    betas = [round(base * m, 4) for m in (0.25, 1.0, 4.0)]
    print(f"\nsuggested: --betas 0.0 {betas[0]} {betas[1]} {betas[2]}")
    print("(0 = unregularized control; the rest span weak to dominant penalty)")


if __name__ == "__main__":
    main()
