"""Demo collection with deliberate quality degradation.

Headroom is the project's single biggest failure risk: if the base policy hits
95% success, every K curve is flat and there is nothing to measure. We therefore
collect from Meta-World's scripted experts with injected action noise and random
episode subsampling, targeting a base policy in the 40-60% band.

Verify headroom BEFORE building any search machinery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ttc.envs.make import DEFAULT_TASKS, make_env


def scripted_policy(task: str):
    from metaworld.policies import ENV_POLICY_MAP  # name varies by version
    key = task.replace("-v2", "").replace("-", " ").title().replace(" ", "")
    return ENV_POLICY_MAP[task]() if task in ENV_POLICY_MAP else None


def collect_task(task, n_episodes, noise_std, out_dir, seed=0, max_steps=500):
    env = make_env(task, seed=seed, max_steps=max_steps)
    expert = scripted_policy(task)
    rng = np.random.default_rng(seed)
    episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10000 + ep)
        O, A, R, D = [], [], [], []
        success = False
        for _ in range(max_steps):
            a = expert.get_action(obs)
            a = np.clip(a + rng.normal(0, noise_std, a.shape), -1, 1)
            O.append(obs); A.append(a)
            obs, r, term, trunc, info = env.step(a)
            R.append(float(r)); D.append(bool(term))
            success = success or bool(info.get("success", 0.0))
            if term or trunc:
                break
        episodes.append(dict(
            obs=np.array(O, np.float32), act=np.array(A, np.float32),
            rew=np.array(R, np.float32), done=np.array(D, bool),
            success=success, task=task,
        ))

    out = Path(out_dir) / f"{task}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, episodes=np.array(episodes, dtype=object))
    rate = np.mean([e["success"] for e in episodes])
    print(f"{task}: {n_episodes} eps, expert success {rate:.2f} -> {out}")
    env.close()
    return rate


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--n-episodes", type=int, default=100)
    p.add_argument("--noise-std", type=float, default=0.3,
                   help="raise to lower base-policy ceiling; tune for 40-60%")
    p.add_argument("--out-dir", default="experiments/data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    for t in a.tasks:
        collect_task(t, a.n_episodes, a.noise_std, a.out_dir, a.seed)
