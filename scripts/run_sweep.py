"""Load checkpoints, assemble selectors, run the paired sweep.

torch>=2.6 defaults torch.load to weights_only=True, which rejects the numpy
arrays we store for observation normalization. These checkpoints are produced by
this repo on this machine, so weights_only=False is safe; the alternative is
allowlisting numpy's reconstructor, which is more fragile across versions.
"""

from __future__ import annotations

import argparse

import torch

from ttc.envs.make import make_env
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
    d = DynamicsEnsemble(ck["obs_dim"], ck["act_dim"], ck["n_members"],
                         hidden=ck.get("hidden", 400))
    sd = dict(ck["state_dict"])
    # dis_mask postdates these checkpoints; derive it from the stored o_std
    # rather than retraining (the mask only affects the uncertainty readout).
    if "dis_mask" not in sd:
        sd["dis_mask"] = (sd["o_std"] > 1e-3).float()
    d.load_state_dict(sd); d.eval()
    # Raise the normalization floor: dims with std < 1e-3 are constant in the
    # data (they normalize to exactly 0 either way), so this changes nothing
    # in-distribution while removing the 1e6x amplification that breaks rollouts.
    d.o_std.clamp_(min=1e-3)
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
    p.add_argument("--plan-horizon", type=int, default=4,
                   help="learned-arm rollout length; past ~4 the dynamics model "
                        "drifts >2 sigma and the value fn bootstraps the tail")
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
        "learned": [LearnedDynamicsSelector(dyn, val, beta=b,
                                           horizon=a.plan_horizon)
                    for b in a.betas],
        "value_bon": [ValueBoNSelector(val, dyn)],
        "consistency": [ConsistencySelector()],
    }
    selectors = [s for arm in a.arms for s in pool[arm]]

    out = run_sweep(lambda seed: make_env(a.task, seed=seed),
                    pol, selectors, task=a.task,
                    n_episodes=a.n_episodes, k_max=a.k_max, out_dir=a.out_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
