"""Outcome verifier: V(s) trained on Monte Carlo returns from demo data.

Used two ways: as a terminal bootstrap for learned-dynamics rollouts, and
standalone as a best-of-N scorer. Standalone is the arm most directly analogous
to reward-model best-of-N in the LLM literature, so it is the one most likely to
show classic over-optimization.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden=256, device="cuda"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        ).to(device)
        self.device = device
        self.register_buffer("o_mean", torch.zeros(obs_dim, device=device))
        self.register_buffer("o_std", torch.ones(obs_dim, device=device))
        self.register_buffer("v_mean", torch.zeros(1, device=device))
        self.register_buffer("v_std", torch.ones(1, device=device))

    def set_normalizer(self, o_mean, o_std, v_mean, v_std):
        for buf, val in [(self.o_mean, o_mean), (self.o_std, o_std),
                         (self.v_mean, v_mean), (self.v_std, v_std)]:
            buf.copy_(torch.as_tensor(val, device=self.device).float().reshape(buf.shape))
        self.o_std.clamp_min_(1e-6); self.v_std.clamp_min_(1e-6)

    def forward(self, obs):
        return self.net((obs - self.o_mean) / self.o_std).squeeze(-1)

    def loss(self, obs, returns):
        return nn.functional.mse_loss(
            self(obs), (returns - self.v_mean) / self.v_std
        )

    @torch.no_grad()
    def value(self, obs):
        return self(obs) * self.v_std + self.v_mean
