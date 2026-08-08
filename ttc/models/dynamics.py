"""Probabilistic dynamics ensemble (PETS-style).

Ensemble disagreement is the load-bearing quantity: it is both the natural
uncertainty signal for regularizing search and, we predict, the thing that
predicts where the K curve turns over. Log it, do not just use it.

Predicts delta-state (not absolute next state) with a learned per-dimension
variance, plus reward and termination.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Member(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=400):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, obs_dim + 1)      # delta-state + reward
        self.logvar = nn.Linear(hidden, obs_dim + 1)
        self.done = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.mu(h), self.logvar(h).clamp(-10, 2), self.done(h)


class DynamicsEnsemble(nn.Module):
    def __init__(self, obs_dim, act_dim, n_members=5, hidden=400, device="cuda"):
        super().__init__()
        self.obs_dim, self.act_dim, self.n = obs_dim, act_dim, n_members
        self.device = device
        self.members = nn.ModuleList(
            [_Member(obs_dim, act_dim, hidden) for _ in range(n_members)]
        ).to(device)
        self.register_buffer("o_mean", torch.zeros(obs_dim, device=device))
        self.register_buffer("o_std", torch.ones(obs_dim, device=device))
        # Meta-World's 39-dim obs contains permanently-constant dims whose std
        # is the 1e-6 clamp floor. Dividing by that inflates the disagreement
        # metric by ~1e6 and lets junk dims dominate. Mask them out of the
        # uncertainty readout; training is unaffected (constant dims map to 0).
        self.register_buffer("dis_mask", torch.ones(obs_dim, device=device))

    def set_normalizer(self, mean, std):
        self.o_mean.copy_(torch.as_tensor(mean, device=self.device).float())
        self.o_std.copy_(torch.as_tensor(std, device=self.device).float().clamp_min(1e-6))
        self.dis_mask.copy_((self.o_std > 1e-3).float())

    def _in(self, obs, act):
        return torch.cat([(obs - self.o_mean) / self.o_std, act], dim=-1)

    def member_loss(self, i, obs, act, next_obs, rew):
        mu, logvar, done = self.members[i](self._in(obs, act))
        target = torch.cat([next_obs - obs, rew[:, None]], dim=-1)
        inv = torch.exp(-logvar)
        # Gaussian NLL; the logvar term is what makes the model report its own
        # uncertainty rather than overconfidently extrapolating.
        return (((mu - target) ** 2) * inv + logvar).mean()

    @torch.no_grad()
    def step(self, obs, act):
        """Batched one-step prediction across the ensemble.

        obs: (B, obs_dim), act: (B, act_dim)
        returns next_obs (n, B, obs_dim), rew (n, B), disagreement (B,)
        """
        x = self._in(obs, act)
        mus = torch.stack([m(x)[0] for m in self.members])       # (n, B, obs+1)
        # Zero the delta on dead dims. They are constant in the data, but the
        # net predicts small nonzero values for them; re-normalizing by the
        # 1e-6 std floor amplifies those ~1e6x into the next step's input,
        # which compounds to NaN over a multi-step rollout.
        next_obs = obs[None] + mus[..., :-1] * self.dis_mask
        rew = mus[..., -1]
        # Disagreement = mean per-dim std across members, normalized by obs scale.
        w = self.dis_mask / self.dis_mask.sum().clamp_min(1)
        disagreement = ((next_obs.std(0) / self.o_std) * w).sum(-1)
        return next_obs, rew, disagreement

    @torch.no_grad()
    def rollout(self, obs, chunks, gamma=0.99):
        """Score action chunks by predicted discounted return.

        obs: (obs_dim,)   chunks: (K, H, act_dim)
        Each ensemble member propagates its own trajectory (no cross-member
        averaging mid-rollout), which is what lets compounding error diverge
        realistically instead of being smoothed away.

        returns ret (K,), disagreement (K,), final_obs (n, K, obs_dim)
        """
        K, H, _ = chunks.shape
        o = obs[None, None].expand(self.n, K, self.obs_dim).clone()
        ret = torch.zeros(self.n, K, device=obs.device)
        dis = torch.zeros(K, device=obs.device)

        for h in range(H):
            a = chunks[None, :, h].expand(self.n, K, self.act_dim)
            flat_o, flat_a = o.reshape(-1, self.obs_dim), a.reshape(-1, self.act_dim)
            x = self._in(flat_o, flat_a)
            mus = torch.stack([m(x)[0] for m in self.members])   # (n, n*K, obs+1)
            # Each member reads its own slice of the flattened batch.
            mus = mus.reshape(self.n, self.n, K, -1)
            mus = mus[torch.arange(self.n), torch.arange(self.n)]  # (n, K, obs+1)
            o = o + mus[..., :-1] * self.dis_mask
            # Belt and braces: keep predicted states inside a plausible range so
            # one bad step cannot poison the remaining horizon.
            lo = self.o_mean - 10 * self.o_std
            hi = self.o_mean + 10 * self.o_std
            o = torch.clamp(o, lo, hi)
            ret += (gamma ** h) * mus[..., -1]
            w = self.dis_mask / self.dis_mask.sum().clamp_min(1)
            dis += ((o.std(0) / self.o_std) * w).sum(-1)

        return ret.mean(0), dis / H, o
