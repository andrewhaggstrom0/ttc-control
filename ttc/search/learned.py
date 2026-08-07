"""Learned-dynamics search, with an explicit uncertainty penalty.

score = predicted discounted return
        + gamma^H * V(predicted final state)      [if value fn supplied]
        - beta * ensemble disagreement            [the regularizer knob]

beta is the key experimental variable. beta=0 is the unregularized version we
expect to over-optimize; sweeping beta tests whether uncertainty penalization
moves or removes the turnover. That sweep is the paper's second figure.
"""

from __future__ import annotations

import numpy as np
import torch

from ttc.search.base import Selector


class LearnedDynamicsSelector(Selector):
    name = "learned"

    def __init__(self, dynamics, value=None, gamma=0.99, beta=0.0,
                 horizon=None, device="cuda"):
        self.dyn, self.value = dynamics, value
        self.gamma, self.beta = gamma, beta
        self.horizon, self.device = horizon, device

    @property
    def name(self):  # beta appears in the run label so sweeps stay separable
        return f"learned_beta{self.beta:g}"

    def _score(self, obs, candidates, env, info):
        o = torch.as_tensor(obs, device=self.device).float().reshape(-1)
        c = torch.as_tensor(candidates, device=self.device).float()
        if self.horizon is not None:
            c = c[:, :self.horizon]

        ret, dis, final = self.dyn.rollout(o, c, gamma=self.gamma)
        score = ret - self.beta * dis

        if self.value is not None:
            v = self.value.value(final.reshape(-1, final.shape[-1]))
            v = v.reshape(final.shape[0], final.shape[1]).mean(0)
            score = score + (self.gamma ** c.shape[1]) * v

        calls = self.dyn.n * c.shape[0] * c.shape[1]
        s = score.cpu().numpy()
        # Stash disagreement for post-hoc analysis: we expect the gap between
        # chosen-candidate disagreement and mean disagreement to widen exactly
        # where the K curve turns over.
        info["disagreement"] = dis.cpu().numpy().tolist()
        return s, calls
