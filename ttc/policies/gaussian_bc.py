"""Deterministic MLP chunk regressor with Gaussian exploration noise.

Positive control for the pipeline. No sampler, no noise schedule, no denoising
loop -- just obs -> action chunk under MSE. If this fails on an easy task, the
bug is in the env/data/eval path, not the generative model.

Also usable as a real ablation policy: sample() adds fixed Gaussian noise so it
produces K distinct candidates for the selectors.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GaussianBCPolicy(nn.Module):
    policy_type = "mlp"

    def __init__(self, obs_dim, act_dim, horizon=16, hidden=512,
                 sample_std=0.15, device="cuda"):
        super().__init__()
        self.obs_dim, self.act_dim, self.horizon = obs_dim, act_dim, horizon
        self.sample_std, self.device = sample_std, device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, horizon * act_dim), nn.Tanh(),
        ).to(device)
        self.register_buffer("obs_mean", torch.zeros(obs_dim, device=device))
        self.register_buffer("obs_std", torch.ones(obs_dim, device=device))

    def set_normalizer(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, device=self.device).float())
        self.obs_std.copy_(
            torch.as_tensor(std, device=self.device).float().clamp_min(1e-6))

    def forward(self, obs):
        h = (obs - self.obs_mean) / self.obs_std
        return self.net(h).reshape(-1, self.horizon, self.act_dim)

    def loss(self, obs, actions):
        return nn.functional.mse_loss(self(obs), actions)

    @torch.no_grad()
    def sample(self, obs, n: int = 1, seed: int | None = None,
               n_denoise: int | None = None) -> np.ndarray:
        """(n, horizon, act_dim). n_denoise accepted and ignored, so this is a
        drop-in for DiffusionPolicy everywhere in the codebase."""
        o = torch.as_tensor(obs, device=self.device).float().reshape(1, -1)
        mean = self(o)                                   # (1, H, A)
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(int(seed))
        noise = torch.randn(n, self.horizon, self.act_dim,
                            device=self.device, generator=g) * self.sample_std
        return (mean + noise).clamp(-1, 1).cpu().numpy()
