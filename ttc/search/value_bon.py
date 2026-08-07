"""Best-of-N with a value function on the predicted immediate outcome.

No dynamics rollout: we score the candidate's first action by V of the state the
value function believes follows. Cheapest verifier, and the closest analogue to
reward-model best-of-N, so it is the natural place for over-optimization to show
up first.
"""

from __future__ import annotations

import numpy as np
import torch

from ttc.search.base import Selector


class ValueBoNSelector(Selector):
    name = "value_bon"

    def __init__(self, value, dynamics, n_steps=1, device="cuda"):
        self.value, self.dyn = value, dynamics
        self.n_steps, self.device = n_steps, device

    def _score(self, obs, candidates, env, info):
        o = torch.as_tensor(obs, device=self.device).float().reshape(1, -1)
        c = torch.as_tensor(candidates, device=self.device).float()
        K = c.shape[0]
        cur = o.expand(K, -1).contiguous()

        for h in range(self.n_steps):
            nxt, _, _ = self.dyn.step(cur, c[:, h])
            cur = nxt.mean(0)

        v = self.value.value(cur)
        return v.cpu().numpy(), self.dyn.n * K * self.n_steps
