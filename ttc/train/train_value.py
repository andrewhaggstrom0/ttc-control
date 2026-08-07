from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ttc.data.dataset import ReturnDataset, load_episodes, obs_normalizer
from ttc.envs.make import DEFAULT_TASKS
from ttc.models.value import ValueFunction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--out", default="experiments/ckpt/value.pt")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    eps = load_episodes(a.data_dir, a.tasks)
    o_mean, o_std = obs_normalizer(eps)
    ds = ReturnDataset(eps, a.gamma)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=2)

    vf = ValueFunction(ds.o.shape[1])
    vf.set_normalizer(o_mean, o_std, ds.g.mean(), ds.g.std())
    opt = torch.optim.AdamW(vf.parameters(), lr=a.lr, weight_decay=1e-5)

    for ep in range(a.epochs):
        tot = 0.0
        for obs, g in dl:
            loss = vf.loss(obs.cuda(), g.cuda())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 20 == 0:
            print(f"epoch {ep} loss {tot / len(dl):.5f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": vf.state_dict(), "obs_dim": ds.o.shape[1]}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
