from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ttc.data.dataset import TransitionDataset, load_episodes, obs_normalizer
from ttc.envs.make import DEFAULT_TASKS
from ttc.models.dynamics import DynamicsEnsemble


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--out", default="experiments/ckpt/dynamics.pt")
    p.add_argument("--n-members", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--frac", type=float, default=1.0,
                   help="subsample data to deliberately degrade model quality; "
                        "sweeping this varies model error independently of K")
    p.add_argument("--hidden", type=int, default=400,
                   help="shrink to degrade the model; subsampling "
                        "alone barely works when 10 percent of the "
                        "data still fits")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    eps = load_episodes(a.data_dir, a.tasks)
    mean, std = obs_normalizer(eps)
    ds = TransitionDataset(eps)
    if a.frac < 1.0:
        n = int(len(ds) * a.frac)
        idx = np.random.default_rng(a.seed).choice(len(ds), n, replace=False)
        ds = Subset(ds, idx)

    obs_dim = eps[0]["obs"].shape[1]
    act_dim = eps[0]["act"].shape[1]
    dyn = DynamicsEnsemble(obs_dim, act_dim, a.n_members, hidden=a.hidden)
    dyn.set_normalizer(mean, std)
    opts = [torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-5)
            for m in dyn.members]

    # Bootstrapped batches: each member sees a different resample, which is what
    # makes ensemble disagreement a meaningful uncertainty signal rather than
    # just init noise.
    # drop_last=True yields ZERO batches when the subsampled dataset is smaller
    # than batch_size, which silently trains nothing and reports nan. Shrink the
    # batch instead of failing quietly.
    bs = min(a.batch_size, max(8, len(ds) // 8))
    if bs < a.batch_size:
        print(f"[warn] {len(ds)} samples: batch {a.batch_size} -> {bs}", flush=True)
    loaders = [DataLoader(ds, batch_size=bs, shuffle=True,
                          num_workers=2, drop_last=True)
               for _ in range(a.n_members)]
    assert len(loaders[0]) > 0, (
        f"empty loader: {len(ds)} samples, batch {bs}. Raise --frac.")

    for ep in range(a.epochs):
        losses = np.zeros(a.n_members)
        for i, dl in enumerate(loaders):
            for o, act, no, r, d in dl:
                o, act, no, r = o.cuda(), act.cuda(), no.cuda(), r.cuda()
                loss = dyn.member_loss(i, o, act, no, r)
                opts[i].zero_grad(); loss.backward(); opts[i].step()
                losses[i] += loss.item()
            losses[i] /= len(dl)
        if ep % 10 == 0:
            print(f"epoch {ep} member losses {np.round(losses, 4)}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": dyn.state_dict(), "obs_dim": obs_dim,
                "act_dim": act_dim, "n_members": a.n_members,
                "hidden": a.hidden, "frac": a.frac}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
