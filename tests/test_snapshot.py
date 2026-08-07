"""Verify MuJoCo state save/restore is exact. The oracle-dynamics arm is
meaningless if branching perturbs the trunk trajectory.

Note: save_state/restore_state live on SnapshotWrapper, NOT on env.unwrapped --
.unwrapped skips every wrapper and returns the bare Sawyer env.
"""
import numpy as np


def test_snapshot_roundtrip(env):
    obs, _ = env.reset(seed=0)
    state = env.save_state()

    for _ in range(10):                       # branch and discard
        env.step(env.action_space.sample())
    env.restore_state(state)

    rng = np.random.RandomState(1)
    trunk_a = [rng.uniform(-1, 1, env.action_space.shape) for _ in range(10)]
    trunk = [env.step(a)[0] for a in trunk_a]

    env.restore_state(state)
    rerun = [env.step(a)[0] for a in trunk_a]

    for t, r in zip(trunk, rerun):
        assert np.array_equal(t, r), "restore is not exact; oracle arm invalid"


def test_snapshot_survives_nested_branches(env):
    """Oracle scores K candidates from one snapshot, restoring between each."""
    env.reset(seed=0)
    state = env.save_state()
    for _ in range(8):
        for _ in range(5):
            env.step(env.action_space.sample())
        env.restore_state(state)
    after = env.save_state()
    assert np.array_equal(state.physics, after.physics)
