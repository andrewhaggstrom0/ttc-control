"""Load checkpoints, assemble selectors, run the paired sweep.

torch>=2.6 defaults torch.load to weights_only=True, which rejects the numpy
arrays we store for observation normalization. These checkpoints are produced by
this repo on this machine, so weights_only=False is safe; the alternative is
allowlisting numpy's reconstructor, which is more fragile across versions.
"""

from __future__ import annotations

import argparse

import torch

from ttc.envs.make import env_fn
from ttc.eval.rollout import run_sweep
from ttc.models.dynamics import DynamicsEnsemble
from ttc.models.value import ValueFunction
from ttc.policies.diffusion import DiffusionPolicy
from ttc.search.base import FirstSelector, RandomSelector
from ttc.search.consistency import ConsistencySelector
from ttc.search.learned import LearnedDynamicsSelector
from ttc.search.oracle import OracleSelector
from ttc.search.value_bon import ValueBoNSelector


def load_policy(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    if ck.get("policy_type") == "mlp":
        from ttc.policies.gaussian_bc import GaussianBCPolicy
        pol = GaussianBCPolicy(ck["obs_dim"], ck["act_dim"], ck["horizon"],
                               sample_std=ck.get("sample_std", 0.15))
        pol.load_state_dict(ck["state_dict"])
        pol.set_normalizer(ck["obs_mean"], ck["obs_std"])
        pol.eval()
        return pol
    pol = DiffusionPolicy(ck["obs_dim"], ck["act_dim"], horizon=ck["horizon"])
    pol.load_state_dict(ck["state_dict"])
    pol.set_normalizer(ck["obs_mean"], ck["obs_std"])
    pol.eval()
    return pol


def load_dynamics(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    d = DynamicsEnsemble(ck["obs_dim"], ck["act_dim"], ck["n_members"])
    d.load_state_dict(ck["state_dict"]); d.eval()
    return d


def load_value(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    v = ValueFunction(ck["obs_dim"]); v.load_state_dict(ck["state_dict"]); v.eval()
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--policy", default="experiments/ckpt/policy.pt")
    p.add_argument("--dynamics", default="experiments/ckpt/dynamics.pt")
    p.add_argument("--value", default="experiments/ckpt/value.pt")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--k-max", type=int, default=64)
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 5.0])
    p.add_argument("--out-dir", default="experiments/rollouts")
    p.add_argument("--arms", nargs="+",
                   default=["first", "random", "oracle", "learned",
                            "value_bon", "consistency"])
    a = p.parse_args()

    pol = load_policy(a.policy)
    dyn = load_dynamics(a.dynamics)
    val = load_value(a.value)

    pool = {
        "first": [FirstSelector()],
        "random": [RandomSelector(seed=0)],
        "oracle": [OracleSelector()],
        "learned": [LearnedDynamicsSelector(dyn, val, beta=b) for b in a.betas],
        "value_bon": [ValueBoNSelector(val, dyn)],
        "consistency": [ConsistencySelector()],
    }
    selectors = [s for arm in a.arms for s in pool[arm]]

    out = run_sweep(env_fn(a.task), pol, selectors, task=a.task,
                    n_episodes=a.n_episodes, k_max=a.k_max, out_dir=a.out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
