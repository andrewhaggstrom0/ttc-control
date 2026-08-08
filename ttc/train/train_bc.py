from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ttc.data.dataset import ChunkDataset, load_episodes, obs_normalizer
from ttc.envs.make import DEFAULT_TASKS
from ttc.policies.gaussian_bc import GaussianBCPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--out", default="experiments/ckpt/policy_bc.pt")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sample-std", type=float, default=0.15)
    a = p.parse_args()

    eps = load_episodes(a.data_dir, a.tasks)
    mean, std = obs_normalizer(eps)
    ds = ChunkDataset(eps, a.horizon)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True)

    obs_dim, act_dim = ds.obs.shape[1], ds.chunks.shape[2]
    pol = GaussianBCPolicy(obs_dim, act_dim, a.horizon, sample_std=a.sample_std)
    pol.set_normalizer(mean, std)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-6)

    for ep in range(a.epochs):
        tot = 0.0
        for obs, chunk in dl:
            loss = pol.loss(obs.cuda(), chunk.cuda())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0:
            print(f"epoch {ep} mse {tot / len(dl):.5f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_type": "mlp", "state_dict": pol.state_dict(),
                "obs_dim": obs_dim, "act_dim": act_dim, "horizon": a.horizon,
                "sample_std": a.sample_std,
                "obs_mean": mean.tolist(), "obs_std": std.tolist()}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
