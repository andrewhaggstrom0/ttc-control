"""Env factory.

Meta-World V3 (current git) renamed all tasks from -v2 to -v3 and registers its
gymnasium namespace at import time. Importing metaworld BEFORE gym.make is
mandatory; skipping it produces a NamespaceNotFound that looks like a missing
install.

Construction is attempted along three paths. Failures are collected rather than
swallowed: a bare `except: pass` here hides which path broke and why, which is
exactly the kind of silent ambiguity that costs days later.

Record the resolved metaworld version in configs/base.yaml. Task names and
observation layout have changed across major versions, so runs collected under
different versions are not comparable.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from ttc.envs.snapshot import SnapshotWrapper

# Spread of difficulty: two easy, two contact-rich, two long-horizon.
DEFAULT_TASKS = (
    "peg-insert-side-v3", "basketball-v3", "stick-pull-v3",
    "shelf-place-v3", "disassemble-v3", "bin-picking-v3",
)


def available_tasks() -> list[str]:
    import metaworld
    from metaworld import _env_dict
    return sorted(_env_dict.ALL_V3_ENVIRONMENTS)


def _build(task: str, seed: int | None):
    import metaworld  # side effect: registers the Meta-World gym namespace

    errors: list[str] = []

    def attempt(label, fn):
        try:
            return fn()
        except Exception as e:
            errors.append(f"  [{label}] {type(e).__name__}: {e}")
            return None

    env = attempt("gym MT1", lambda: gym.make(
        "Meta-World/MT1", env_name=task, seed=seed, disable_env_checker=True))
    if env is not None:
        return env

    env = attempt("gym direct", lambda: gym.make(
        f"Meta-World/{task}", seed=seed, disable_env_checker=True))
    if env is not None:
        return env

    def legacy():
        mt1 = metaworld.MT1(task, seed=seed or 0)
        e = mt1.train_classes[task]()
        e.set_task(mt1.train_tasks[0])
        return e

    env = attempt("metaworld MT1", legacy)
    if env is not None:
        return env

    known = available_tasks()
    hint = ""
    if task not in known:
        near = [t for t in known if task.split("-")[0] in t]
        hint = f"\n'{task}' is not a known task. Did you mean: {near[:5]}"
    raise RuntimeError(
        f"could not construct env '{task}'; all paths failed:\n"
        + "\n".join(errors) + hint
    )


def make_env(task: str, seed: int | None = None, max_steps: int = 500):
    env = _build(task, seed)
    if not any(isinstance(w, gym.wrappers.TimeLimit) for w in _wrapper_chain(env)):
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    return SnapshotWrapper(env)


def _wrapper_chain(env):
    while isinstance(env, gym.Wrapper):
        yield env
        env = env.env


def env_fn(task: str, seed: int | None = None, max_steps: int = 500):
    """Thunk for run_sweep, which wants a fresh env per episode."""
    return lambda: make_env(task, seed, max_steps)


def obs_action_dims(task: str) -> tuple[int, int]:
    e = make_env(task)
    d = (int(np.prod(e.observation_space.shape)), int(np.prod(e.action_space.shape)))
    e.close()
    return d
