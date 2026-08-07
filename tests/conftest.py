"""Shared fixtures.

The env fixture is deliberately a cheap task: snapshot correctness is a property
of the wrapper, not the task, and reach-v3 resets fast enough to keep the test
in the seconds range.
"""

import pytest

from ttc.envs.make import make_env


@pytest.fixture
def env():
    e = make_env("reach-v3", seed=0, max_steps=100)
    yield e
    e.close()
