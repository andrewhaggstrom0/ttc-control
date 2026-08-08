"""Demo collection with deliberate quality degradation.

Headroom is the project's single biggest failure risk: if the base policy hits
95% success, every K curve is flat and there is nothing to measure. We collect
from Meta-World's scripted experts with injected action noise, targeting a base
policy in the 40-60% band.

Policy lookup is by discovery, not by importing a fixed symbol: the map's name
has moved across Meta-World versions (ENV_POLICY_MAP, _POLICIES, etc.), so we
try the map first and fall back to reconstructing the class name from the task.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from ttc.envs.make import DEFAULT_TASKS, make_env


def _class_name(task: str) -> str:
    """'peg-insert-side-v3' -> 'SawyerPegInsertSideV3Policy'"""
    parts = re.split(r"[-_]", task)
    return "Sawyer" + "".join(p.capitalize() for p in parts) + "Policy"


def scripted_policy(task: str):
    import metaworld.policies as P

    for attr in ("ENV_POLICY_MAP", "_ENV_POLICY_MAP", "POLICY_MAP"):
        m = getattr(P, attr, None)
        if isinstance(m, dict) and task in m:
            return m[task]()

    cls = getattr(P, _class_name(task), None)
    if cls is not None:
        return cls()

    # Last resort: case-insensitive match on the de-punctuated task name.
    key = re.sub(r"[-_]", "", task).lower()
    for n in dir(P):
        if n.lower().startswith("sawyer") and key in n.lower().replace("policy", ""):
            return getattr(P, n)()

    raise KeyError(
        f"no scripted policy found for '{task}'. Tried maps and "
        f"'{_class_name(task)}'. Available: "
        f"{[n for n in dir(P) if n.startswith('Sawyer')][:10]}"
    )


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
            a = np.asarray(expert.get_action(obs), dtype=np.float32)
            a = np.clip(a + rng.normal(0, noise_std, a.shape), -1, 1)
            O.append(obs); A.append(a)
            obs, r, term, trunc, info = env.step(a)
            R.append(float(r)); D.append(bool(term))
            if bool(info.get("success", 0.0)):
                success = True
                break          # stop at first success: length becomes a real
                               # difficulty signal, and we stop recording
                               # hundreds of post-success steps as "expert" data
            if term or trunc:
                break
        episodes.append(dict(
            obs=np.array(O, np.float32), act=np.array(A, np.float32),
            rew=np.array(R, np.float32), done=np.array(D, bool),
            success=success, task=task, noise_std=noise_std,
        ))

    out = Path(out_dir) / f"{task}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, episodes=np.array(episodes, dtype=object))
    rate = float(np.mean([e["success"] for e in episodes]))
    lens = np.mean([len(e["obs"]) for e in episodes])
    print(f"{task:24s} n={n_episodes:4d}  expert_success={rate:.2f}  "
          f"mean_len={lens:5.1f}  -> {out}", flush=True)
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

    rates = {t: collect_task(t, a.n_episodes, a.noise_std, a.out_dir, a.seed)
             for t in a.tasks}
    print("\nexpert success under noise:")
    for t, r in sorted(rates.items(), key=lambda kv: kv[1]):
        print(f"  {t:24s} {r:.2f}")
