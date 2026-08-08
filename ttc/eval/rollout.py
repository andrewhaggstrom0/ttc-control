"""Closed-loop evaluation with per-step selection logging.

Design intent: one pass produces the data for every K in the sweep. We sample
K_max candidates per decision point and evaluate nested prefixes
(K=1,2,4,...,K_max), so curves across K are paired on identical candidates and
identical environment stochasticity.

Everything needed for post-hoc analysis is written to disk per step. Do not
reduce here; reduce in eval/metrics.py. The expensive thing is the rollout, and
you will think of new statistics after you have run it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ttc.search.base import Selector, candidate_seed


@dataclass
class StepRecord:
    step: int
    k: int
    chosen_index: int
    chosen_score: float
    score_spread: float
    scores: list[float]
    reward: float
    n_model_calls: int
    select_time_s: float


@dataclass
class EpisodeRecord:
    episode_id: int
    task: str
    selector: str
    k: int
    seed: int
    success: bool
    total_reward: float
    length: int
    wall_time_s: float
    steps: list[StepRecord] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d["steps"] = [asdict(s) if not isinstance(s, dict) else s for s in self.steps]
        return json.dumps(d)


def k_schedule(k_max: int) -> list[int]:
    """Nested powers of two up to k_max, always including k_max itself."""
    ks, k = [], 1
    while k < k_max:
        ks.append(k)
        k *= 2
    ks.append(k_max)
    return ks


def run_episode(
    env,
    policy,
    selector: Selector,
    *,
    episode_id: int,
    k: int,
    task: str,
    max_steps: int = 500,
    n_exec: int = 8,
    seed: int | None = None,
    candidate_cache: dict[int, np.ndarray] | None = None,
) -> EpisodeRecord:
    """Run one closed-loop episode under a given selector and K.

    n_exec: how many actions of the chosen chunk to execute before replanning.
      This is a second, independent compute axis (replan frequency). Hold it
      fixed at 8 for the main sweep; sweep it separately later.

    candidate_cache: optional dict mapping step -> (K_max, H, A). Pass the same
      cache across selectors to guarantee paired candidates. If None, candidates
      are regenerated from candidate_seed(), which is deterministic anyway.
    """
    ep_seed = seed if seed is not None else episode_id
    obs, _ = env.reset(seed=ep_seed)

    rec = EpisodeRecord(
        episode_id=episode_id, task=task, selector=selector.name, k=k,
        seed=ep_seed, success=False, total_reward=0.0, length=0, wall_time_s=0.0,
    )
    t_start = time.perf_counter()

    step = 0
    while step < max_steps:
        if candidate_cache is not None and step in candidate_cache:
            pool = candidate_cache[step]
        else:
            pool = policy.sample(obs, n=k, seed=candidate_seed(episode_id, step))
            if candidate_cache is not None:
                candidate_cache[step] = pool
        candidates = pool[:k]          # nested prefix

        result = selector.select(obs, candidates, env=env, info={"step": step})
        chunk = candidates[result.index]

        chunk_reward, terminated, truncated, success = 0.0, False, False, False
        for a in chunk[:n_exec]:
            obs, r, terminated, truncated, info = env.step(a)
            chunk_reward += float(r)
            # Meta-World reports success in info; treat it as latching, since a
            # task solved mid-chunk should count even if the arm drifts after.
            success = success or bool(info.get("success", 0.0))
            step += 1
            if terminated or truncated or step >= max_steps:
                break

        rec.steps.append(StepRecord(
            step=step, k=k, chosen_index=result.index,
            chosen_score=result.chosen_score, score_spread=result.score_spread,
            scores=result.scores.tolist(), reward=chunk_reward,
            n_model_calls=result.n_model_calls, select_time_s=result.wall_time_s,
        ))
        rec.total_reward += chunk_reward
        rec.success = rec.success or success

        if terminated or truncated or rec.success:
            break

    rec.length = step
    rec.wall_time_s = time.perf_counter() - t_start
    return rec


def run_sweep(
    env_builder,
    policy,
    selectors: Sequence[Selector],
    *,
    task: str,
    n_episodes: int,
    k_max: int,
    out_dir: str | Path,
    max_steps: int = 500,
    n_exec: int = 8,
) -> Path:
    """Full paired sweep: every selector x every nested K x every episode.

    Episode ordering is outermost so that a killed job still leaves complete,
    balanced data for the episodes it finished. Results append to JSONL as they
    land; nothing is held in memory across episodes.

    env_builder takes a SEED, not zero args. Meta-World's RandomTaskSelectWrapper
    draws a goal configuration at CONSTRUCTION time, so building the env without
    a seed gives every (selector, K) a different task instance and destroys the
    pairing the whole design depends on. Symptom when this is wrong: `first`
    success varies with K even though it always returns candidate 0.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_rollouts.jsonl"
    ks = k_schedule(k_max)

    with out_path.open("a") as f:
        for ep in range(n_episodes):
            # One candidate pool per episode, shared by all selectors and all K.
            # Note this is only exactly shared while trajectories agree; once
            # selectors diverge the observations differ and pooling by step is an
            # approximation. Keep it for K-pairing within a selector, which is
            # exact, and treat cross-selector pairing as partial.
            for sel in selectors:
                cache: dict[int, np.ndarray] = {}
                for k in ks:
                    env = env_builder(ep)   # same seed -> same task instance
                    try:
                        rec = run_episode(
                            env, policy, sel, episode_id=ep, k=k, task=task,
                            max_steps=max_steps, n_exec=n_exec,
                            candidate_cache=cache if k == k_max else None,
                        )
                    finally:
                        env.close()
                    f.write(rec.to_json() + "\n")
                    f.flush()
    return out_path
