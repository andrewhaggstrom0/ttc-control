"""Oracle-dynamics search: branch the real simulator.

This is the upper bound that makes the whole design interpretable. If success
rises monotonically here but turns over under the learned model, the turnover is
attributable to model error rather than to search being useless.

Correctness requirement: state must be restored exactly after every branch. The
assertion below is cheap relative to simulation cost -- leave it on.
"""

from __future__ import annotations

import numpy as np

from ttc.search.base import Selector


class OracleSelector(Selector):
    name = "oracle"
    requires_env = True

    def __init__(self, gamma: float = 0.99, verify: bool = True,
                 success_bonus: float = 10.0):
        self.gamma = gamma
        self.verify = verify
        self.success_bonus = success_bonus

    def _score(self, obs, candidates, env, info):
        snap = env.save_state()
        K, H, _ = candidates.shape
        scores = np.zeros(K)
        calls = 0

        for k in range(K):
            total, disc = 0.0, 1.0
            for h in range(H):
                _, r, term, trunc, step_info = env.step(candidates[k, h])
                total += disc * float(r)
                if step_info.get("success", 0.0):
                    total += disc * self.success_bonus
                disc *= self.gamma
                calls += 1
                if term or trunc:
                    break
            scores[k] = total
            env.restore_state(snap)

        if self.verify:
            after = env.save_state()
            assert np.array_equal(snap.physics, after.physics), \
                "oracle branching corrupted trunk state"

        return scores, calls
