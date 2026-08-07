"""Datasets for the three trainable components.

ChunkDataset  -> policy   (obs, action-chunk)
TransitionDataset -> dynamics (obs, act, next_obs, rew, done)
ReturnDataset -> value    (obs, discounted return-to-go)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_episodes(data_dir, tasks):
    eps = []
    for t in tasks:
        f = Path(data_dir) / f"{t}.npz"
        if f.exists():
            eps.extend(np.load(f, allow_pickle=True)["episodes"].tolist())
    if not eps:
        raise FileNotFoundError(f"no episode files in {data_dir} for {tasks}")
    return eps


def obs_normalizer(episodes):
    obs = np.concatenate([e["obs"] for e in episodes], 0)
    return obs.mean(0), obs.std(0) + 1e-6


class ChunkDataset(Dataset):
    def __init__(self, episodes, horizon=16):
        self.h = horizon
        self.obs, self.chunks = [], []
        for e in episodes:
            T = len(e["obs"])
            for t in range(T):
                a = e["act"][t:t + horizon]
                if len(a) < horizon:  # pad by repeating final action
                    a = np.concatenate([a, np.repeat(a[-1:], horizon - len(a), 0)])
                self.obs.append(e["obs"][t]); self.chunks.append(a)
        self.obs = np.array(self.obs, np.float32)
        self.chunks = np.array(self.chunks, np.float32)

    def __len__(self): return len(self.obs)

    def __getitem__(self, i):
        return torch.from_numpy(self.obs[i]), torch.from_numpy(self.chunks[i])


class TransitionDataset(Dataset):
    def __init__(self, episodes):
        o, a, no, r, d = [], [], [], [], []
        for e in episodes:
            T = len(e["obs"])
            o.append(e["obs"][:T - 1]); a.append(e["act"][:T - 1])
            no.append(e["obs"][1:]); r.append(e["rew"][:T - 1]); d.append(e["done"][:T - 1])
        self.o = np.concatenate(o).astype(np.float32)
        self.a = np.concatenate(a).astype(np.float32)
        self.no = np.concatenate(no).astype(np.float32)
        self.r = np.concatenate(r).astype(np.float32)
        self.d = np.concatenate(d).astype(np.float32)

    def __len__(self): return len(self.o)

    def __getitem__(self, i):
        return tuple(torch.from_numpy(x[i]) for x in (self.o, self.a, self.no)) + \
               (torch.tensor(self.r[i]), torch.tensor(self.d[i]))


class ReturnDataset(Dataset):
    def __init__(self, episodes, gamma=0.99):
        o, g = [], []
        for e in episodes:
            R, acc = np.zeros(len(e["rew"]), np.float32), 0.0
            for t in reversed(range(len(e["rew"]))):
                acc = e["rew"][t] + gamma * acc
                R[t] = acc
            o.append(e["obs"]); g.append(R)
        self.o = np.concatenate(o).astype(np.float32)
        self.g = np.concatenate(g).astype(np.float32)

    def __len__(self): return len(self.o)

    def __getitem__(self, i):
        return torch.from_numpy(self.o[i]), torch.tensor(self.g[i])
