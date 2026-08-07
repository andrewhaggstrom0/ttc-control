"""Self-consistency: no verifier at all.

Pick the medoid of the candidate set -- the chunk closest to all the others.
This is the sleeper arm. Having no learned scorer, it has nothing to Goodhart,
so if it holds up at large K while verifier-based selection turns over, that is
both the cleanest evidence for the over-optimization story and the most
practically useful result in the paper.
"""

from __future__ import annotations

import numpy as np

from ttc.search.base import Selector


class ConsistencySelector(Selector):
    name = "consistency"

    def __init__(self, weight_decay: float = 1.0, first_step_only: bool = False):
        # weight_decay < 1 downweights later timesteps, on the theory that
        # agreement about the immediate action matters more than about the tail.
        self.weight_decay = weight_decay
        self.first_step_only = first_step_only

    def _score(self, obs, candidates, env, info):
        c = candidates[:, :1] if self.first_step_only else candidates
        K, H, A = c.shape
        w = self.weight_decay ** np.arange(H)
        w = (w / w.sum())[None, None, :, None]

        diff = c[:, None] - c[None, :]                  # (K, K, H, A)
        d = np.sqrt(((diff ** 2) * w).sum(axis=(2, 3)))  # (K, K)
        return -d.mean(axis=1), 0
