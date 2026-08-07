"""Exact MuJoCo state save/restore across the full wrapper chain.

Two independent things must be restored:

  1. MuJoCo physics. We use mjSTATE_INTEGRATION, NOT mjSTATE_FULLPHYSICS:
     FULLPHYSICS omits qacc_warmstart, the solver's cached initial guess. A
     discarded branch leaves a different warmstart than the trunk, so the
     constraint solver converges along a slightly different path and the
     roundtrip differs in the last bits. INTEGRATION covers physics plus
     warmstart, ctrl, applied forces, mocap, userdata, and eq_active.

  2. Python-side state on EVERY wrapper, not just the unwrapped env. Meta-World
     V3 via gym.make stacks TimeLimit, AutoTerminateOnSuccess,
     RecordEpisodeStatistics, RandomTaskSelect, CheckpointWrapper, and
     OrderEnforcing. Each holds mutable counters or caches. The base env also
     caches _prev_obs, which feeds observation indices 18-21 -- restoring
     physics alone leaves those stale and the roundtrip test fails there.

An allowlist of attribute names does not survive version changes. We instead
deep-copy every non-excluded instance attribute along the chain, discovering the
copyable key set once at construction.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import ModuleType

import gymnasium as gym
import mujoco
import numpy as np

# Never copy: the env graph itself, spaces, MuJoCo handles, renderers.
_EXCLUDE_TYPES = (gym.Env, gym.Space, mujoco.MjModel, mujoco.MjData, ModuleType)
_EXCLUDE_KEYS = frozenset({
    "env", "unwrapped", "model", "data", "mj_model", "mj_data",
    "_model", "_data", "sim", "renderer", "mujoco_renderer",
    "observation_space", "action_space", "_observation_space", "_action_space",
    "spec", "metadata",
})


@dataclass(frozen=True)
class MjSnapshot:
    physics: np.ndarray
    ctrl: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    userdata: np.ndarray
    warmstart: np.ndarray
    py_state: tuple[dict, ...]   # one dict per level of the wrapper chain


class SnapshotWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._model = self._find(("model", "mj_model", "_model"))
        self._data = self._find(("data", "mj_data", "_data"))
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._size = mujoco.mj_stateSize(self._model, self._spec)
        self._chain = list(self._walk())
        self._keys = [self._copyable_keys(o) for o in self._chain]

    def _walk(self):
        """Every object below this wrapper, outermost first."""
        e = self.env
        while True:
            yield e
            if not isinstance(e, gym.Wrapper):
                break
            e = e.env

    def _find(self, names):
        for obj in self._walk() if hasattr(self, "_chain") else [self.env.unwrapped]:
            for n in names:
                v = getattr(obj, n, None)
                if isinstance(v, (mujoco.MjModel, mujoco.MjData)):
                    return v
        for n in names:
            v = getattr(self.env.unwrapped, n, None)
            if v is not None:
                return v
        raise AttributeError(f"could not locate MuJoCo handle among {names}")

    @staticmethod
    def _copyable_keys(obj) -> tuple[str, ...]:
        keys = []
        for k, v in vars(obj).items():
            if k in _EXCLUDE_KEYS or isinstance(v, _EXCLUDE_TYPES) or callable(v):
                continue
            try:
                copy.deepcopy(v)
            except Exception:
                continue
            keys.append(k)
        return tuple(keys)

    def save_state(self) -> MjSnapshot:
        buf = np.empty(self._size, dtype=np.float64)
        mujoco.mj_getState(self._model, self._data, buf, self._spec)
        py = tuple(
            {k: copy.deepcopy(getattr(obj, k)) for k in keys if hasattr(obj, k)}
            for obj, keys in zip(self._chain, self._keys)
        )
        return MjSnapshot(
            physics=buf,
            ctrl=self._data.ctrl.copy(),
            mocap_pos=self._data.mocap_pos.copy(),
            mocap_quat=self._data.mocap_quat.copy(),
            userdata=self._data.userdata.copy(),
            warmstart=self._data.qacc_warmstart.copy(),
            py_state=py,
        )

    def restore_state(self, snap: MjSnapshot) -> None:
        # INTEGRATION already carries ctrl/mocap/userdata; the fields on
        # MjSnapshot are retained for diagnostics only.
        mujoco.mj_setState(self._model, self._data, snap.physics, self._spec)
        for obj, state in zip(self._chain, snap.py_state):
            for k, v in state.items():
                setattr(obj, k, copy.deepcopy(v))
        # mj_forward is required so the first post-restore step sees correct
        # forward kinematics (Meta-World's obs reads site/body positions). But
        # it runs the constraint solver and OVERWRITES qacc_warmstart as a side
        # effect, silently undoing part of the restore. Reapply it afterwards --
        # without this, nested branch-and-restore cycles drift.
        mujoco.mj_forward(self._model, self._data)
        self._data.qacc_warmstart[:] = snap.warmstart

    def diff_after_steps(self, n: int = 5) -> dict[str, list[str]]:
        """Diagnostic: which attributes mutate during stepping, by chain level.
        Run this after a Meta-World upgrade to see what changed."""
        self.reset(seed=0)
        before = self.save_state()
        for _ in range(n):
            self.step(self.action_space.sample())
        after = self.save_state()
        out = {}
        for obj, b, a in zip(self._chain, before.py_state, after.py_state):
            changed = [k for k in b if repr(b[k]) != repr(a.get(k))]
            if changed:
                out[type(obj).__name__] = changed
        return out
