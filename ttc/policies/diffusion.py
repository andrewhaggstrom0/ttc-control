"""Diffusion policy over action chunks.

Two test-time compute knobs live here and must not be confused:
  n            -- parallel samples (the K axis; consumed by selectors)
  n_denoise    -- sequential refinement steps (an independent axis)

sample() is seeded explicitly so candidate sets are reproducible and paired
across selectors, per the contract in search/base.py.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None].float() * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ConditionalUnet1D(nn.Module):
    """Deliberately small MLP-style denoiser. State-based obs means we do not
    need a real U-Net, and a smaller net keeps base performance in the 40-60%
    headroom band where search has something to find."""

    def __init__(self, obs_dim, act_dim, horizon, hidden=512, t_dim=64):
        super().__init__()
        self.horizon, self.act_dim = horizon, act_dim
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(horizon * act_dim + obs_dim + t_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, horizon * act_dim),
        )

    def forward(self, x, t, obs):
        B = x.shape[0]
        h = torch.cat([x.reshape(B, -1), obs, _sinusoidal(t, self.t_dim)], dim=-1)
        return self.net(h).reshape(B, self.horizon, self.act_dim)


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon=16, n_train_steps=100,
                 n_denoise=100, device="cuda"):
        super().__init__()
        self.obs_dim, self.act_dim, self.horizon = obs_dim, act_dim, horizon
        self.n_train_steps, self.n_denoise = n_train_steps, n_denoise
        self.device = device
        self.net = ConditionalUnet1D(obs_dim, act_dim, horizon).to(device)

        # Squared-cosine schedule (Nichol & Dhariwal; used by Diffusion Policy).
        # NOT linspace(1e-4, 0.02): those endpoints are DDPM's and assume 1000
        # steps. Over 100 steps they leave alphas_cum[-1] ~= 0.37, so training
        # never sees near-pure noise while sampling starts from pure noise --
        # a train/sample mismatch that caps the model far below usable quality.
        t = torch.linspace(0, 1, n_train_steps + 1, device=device)
        f = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cum = (f / f[0])[1:]
        betas = (1 - alphas_cum / torch.cat([f[:1] / f[0], alphas_cum[:-1]]))
        betas = betas.clamp(max=0.999)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cum", alphas_cum)
        self.register_buffer("obs_mean", torch.zeros(obs_dim, device=device))
        self.register_buffer("obs_std", torch.ones(obs_dim, device=device))

    def set_normalizer(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, device=self.device).float())
        self.obs_std.copy_(torch.as_tensor(std, device=self.device).float().clamp_min(1e-6))

    def _norm(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def loss(self, obs, actions):
        B = obs.shape[0]
        t = torch.randint(0, self.n_train_steps, (B,), device=self.device)
        noise = torch.randn_like(actions)
        a_cum = self.alphas_cum[t][:, None, None]
        noisy = a_cum.sqrt() * actions + (1 - a_cum).sqrt() * noise
        pred = self.net(noisy, t, self._norm(obs))
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, obs, n: int = 1, seed: int | None = None,
               n_denoise: int | None = None) -> np.ndarray:
        """Return (n, horizon, act_dim) in [-1, 1]."""
        steps = n_denoise or self.n_denoise
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(int(seed))

        o = torch.as_tensor(obs, device=self.device).float().reshape(1, -1)
        o = self._norm(o).expand(n, -1)
        x = torch.randn(n, self.horizon, self.act_dim,
                        device=self.device, generator=g)

        # Strided DDIM-style schedule so n_denoise is a real compute knob.
        idx = torch.linspace(self.n_train_steps - 1, 0, steps).long().to(self.device)
        for i, t in enumerate(idx):
            tb = t.repeat(n)
            eps = self.net(x, tb, o)
            a_t = self.alphas_cum[t]
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            if i < len(idx) - 1:
                a_prev = self.alphas_cum[idx[i + 1]]
                x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
            else:
                x = x0
        return x.cpu().numpy()
