from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ttc.data.dataset import ChunkDataset, load_episodes, obs_normalizer
from ttc.envs.make import DEFAULT_TASKS
from ttc.policies.diffusion import DiffusionPolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--out", default="experiments/ckpt/policy.pt")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    eps = load_episodes(a.data_dir, a.tasks)
    mean, std = obs_normalizer(eps)
    ds = ChunkDataset(eps, a.horizon)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                    num_workers=4, drop_last=True)

    obs_dim, act_dim = ds.obs.shape[1], ds.chunks.shape[2]
    pol = DiffusionPolicy(obs_dim, act_dim, horizon=a.horizon)
    pol.set_normalizer(mean, std)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-6)
    ema = torch.optim.swa_utils.AveragedModel(
        pol, avg_fn=lambda e, c, n: 0.995 * e + 0.005 * c)

    for ep in range(a.epochs):
        tot = 0.0
        for obs, chunk in dl:
            obs, chunk = obs.cuda(), chunk.cuda()
            loss = pol.loss(obs, chunk)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step(); ema.update_parameters(pol)
            tot += loss.item()
        if ep % 10 == 0:
            print(f"epoch {ep} loss {tot / len(dl):.5f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ema.module.state_dict(), "obs_dim": obs_dim,
                "act_dim": act_dim, "horizon": a.horizon,
                "obs_mean": mean, "obs_std": std}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
