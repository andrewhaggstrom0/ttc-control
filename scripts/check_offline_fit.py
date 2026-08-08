"""Does the policy reproduce demo actions on demo states?

Distinguishes two failure modes that look identical from success rate alone:
  - high L1  -> underfitting / mode averaging; the policy never learned the data
  - low L1 but failing rollouts -> covariate shift; it fits states it saw and
    diverges once its own errors take it off-distribution

Also reports whether sampled actions match the demos in scale, which catches
normalization and clipping bugs.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ttc.data.dataset import ChunkDataset, load_episodes
from ttc.envs.make import DEFAULT_TASKS
from scripts.run_sweep import load_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--policy", default="experiments/ckpt/policy.pt")
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--n-samples", type=int, default=300)
    p.add_argument("--n-per-state", type=int, default=8)
    a = p.parse_args()

    pol = load_policy(a.policy)
    eps = load_episodes(a.data_dir, a.tasks)
    ds = ChunkDataset(eps, pol.horizon)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), min(a.n_samples, len(ds)), replace=False)

    l1_best, l1_mean, spreads = [], [], []
    for i in idx:
        obs, true_chunk = ds[int(i)]
        cand = pol.sample(obs.numpy(), n=a.n_per_state, seed=int(i))
        err = np.abs(cand - true_chunk.numpy()).mean(axis=(1, 2))
        l1_best.append(err.min())
        l1_mean.append(err.mean())
        spreads.append(cand.std(axis=0).mean())

    demo_acts = np.concatenate([e["act"] for e in eps])
    samp = pol.sample(ds[int(idx[0])][0].numpy(), n=64, seed=0)

    print(f"demo action   mean={demo_acts.mean():+.3f} std={demo_acts.std():.3f} "
          f"min={demo_acts.min():+.2f} max={demo_acts.max():+.2f}")
    print(f"policy sample mean={samp.mean():+.3f} std={samp.std():.3f} "
          f"min={samp.min():+.2f} max={samp.max():+.2f}")
    print(f"\nchunk L1 vs demo  best-of-{a.n_per_state}={np.mean(l1_best):.4f}"
          f"  mean={np.mean(l1_mean):.4f}")
    print(f"candidate spread (std across samples) = {np.mean(spreads):.4f}")
    print("\ninterpretation:")
    print("  L1 > ~0.3        -> underfitting or mode averaging")
    print("  L1 < ~0.1        -> fits offline; failure is closed-loop drift")
    print("  spread ~0        -> mode collapse; search has nothing to choose from")
    print("  spread > ~0.5    -> too diffuse; candidates are near-random")


if __name__ == "__main__":
    main()
