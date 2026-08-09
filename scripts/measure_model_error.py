"""Measure multi-step dynamics error independently of any search.

Gives you the x-axis for the dose-response plot: model error vs how fast search
degrades. Without this you have "search with a bad model hurts"; with it you
have a quantitative relationship.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ttc.envs.make import make_env
from scripts.run_sweep import load_dynamics, load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--dynamics", required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--n-episodes", type=int, default=5)
    a = p.parse_args()

    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    dyn = load_dynamics(a.dynamics)
    errs = []

    for ep in range(a.n_episodes):
        env = make_env(a.task, seed=ep)
        obs, _ = env.reset(seed=ep)
        for start in range(0, 160, 20):
            chunk = pol.sample(obs, n=1, seed=ep * 100 + start)[0]
            snap = env.save_state()
            true = []
            for act in chunk[:a.horizon]:
                o2, _, term, trunc, _ = env.step(act)
                true.append(o2)
                if term or trunc:
                    break
            env.restore_state(snap)
            if len(true) < a.horizon:
                break
            o = torch.as_tensor(obs, device="cuda").float().reshape(-1)
            ct = torch.as_tensor(chunk[None, :a.horizon], device="cuda").float()
            _, _, final = dyn.rollout(o, ct)
            pred = final.mean(0)[0]
            gt = torch.as_tensor(true[-1], device="cuda").float()
            errs.append(((pred - gt).abs() / dyn.o_std).max().item())
            for act in chunk[:8]:
                obs, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
            if term or trunc:
                break
        env.close()

    e = np.array(errs)
    print(f"{a.task}\t{a.dynamics}\th={a.horizon}\t"
          f"median={np.median(e):.3f}\tmean={e.mean():.3f}\tp90={np.percentile(e,90):.3f}")


if __name__ == "__main__":
    main()
