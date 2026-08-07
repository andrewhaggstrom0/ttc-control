"""Selector interface.

Every test-time-compute mechanism in this project is a Selector: it receives a
fixed set of candidate action chunks and returns which one to execute, plus the
scores it assigned. Keeping this interface narrow is what lets the K sweep be a
config change rather than four divergent codepaths.

Contract notes:
  - Selectors MUST NOT mutate `env`. Oracle-style selectors that branch the
    simulator are responsible for restoring exact state before returning; see
    tests/test_snapshot.py for the invariant they rely on.
  - `scores` is persisted verbatim by the eval loop. It is the raw material for
    the verifier-vs-outcome correlation, which is the headline analysis. Return
    real numbers even when the selector is trivial.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def candidate_seed(episode_id: int, step: int, salt: str = "cand") -> int:
    """Deterministic seed for candidate sampling.

    Identical (episode_id, step) yields identical candidates regardless of which
    selector is running or what K is. This makes selector comparisons paired,
    which matters enormously at the episode counts we can afford.
    """
    key = f"{salt}:{episode_id}:{step}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


@dataclass
class SelectionResult:
    index: int                                  # which candidate to execute
    scores: np.ndarray                          # (K,) score per candidate
    n_model_calls: int = 0                      # learned-model or sim forward passes
    wall_time_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chosen_score(self) -> float:
        return float(self.scores[self.index])

    @property
    def score_spread(self) -> float:
        """Max-minus-median. A cheap proxy for how much selection pressure was
        applied this step; expected to grow with K and to correlate with
        over-optimization once the verifier is imperfect."""
        if self.scores.size < 2:
            return 0.0
        return float(self.scores.max() - np.median(self.scores))


class Selector(ABC):
    """Base class. Subclasses implement `_score`, not `select`."""

    name: str = "base"
    requires_env: bool = False   # True for oracle: needs a branchable simulator

    @abstractmethod
    def _score(
        self,
        obs: np.ndarray,
        candidates: np.ndarray,   # (K, H, A)
        env: Any | None,
        info: dict[str, Any],
    ) -> tuple[np.ndarray, int]:
        """Return (scores of shape (K,), number of model calls consumed)."""

    def select(
        self,
        obs: np.ndarray,
        candidates: np.ndarray,
        env: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> SelectionResult:
        if self.requires_env and env is None:
            raise ValueError(f"{self.name} requires an env handle to branch from")
        if candidates.ndim != 3:
            raise ValueError(f"expected candidates (K, H, A), got {candidates.shape}")

        t0 = time.perf_counter()
        scores, n_calls = self._score(obs, candidates, env, info or {})
        elapsed = time.perf_counter() - t0

        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != candidates.shape[0]:
            raise ValueError("scores length must equal number of candidates")
        if not np.all(np.isfinite(scores)):
            # Do not silently nan-to-num. A non-finite score means the dynamics
            # model diverged, which is itself a finding worth surfacing loudly.
            raise FloatingPointError(f"{self.name} produced non-finite scores")

        return SelectionResult(
            index=int(np.argmax(scores)),
            scores=scores,
            n_model_calls=n_calls,
            wall_time_s=elapsed,
            meta={"selector": self.name},
        )


class FirstSelector(Selector):
    """K=1 baseline: execute the policy's first sample, no search.

    Scores are constant, so chosen_score carries no information and the
    correlation analysis correctly reports undefined for this arm.
    """

    name = "first"

    def _score(self, obs, candidates, env, info):
        return np.zeros(candidates.shape[0]), 0


class RandomSelector(Selector):
    """Control condition. Isolates 'more samples' from 'better selection': any
    gain here is diversity, not verification. Worth running once at each K."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def _score(self, obs, candidates, env, info):
        return self.rng.random(candidates.shape[0]), 0
