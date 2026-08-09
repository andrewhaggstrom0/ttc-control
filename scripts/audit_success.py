"""Audit candidate picks against EVENTUAL SUCCESS, not 16-step return.

scripts/audit_selection.py showed regret ~1.0 against oracle score, implying the
learned verifier picks an average candidate. But the learned arm's success
(0.30) is far below the random arm's (0.82) on the same task, so "average by
16-step return" is clearly not "average by task outcome".

Here we execute each candidate and then continue to episode completion under the
base policy, recording success. That measures what a candidate is actually worth.
Expensive: K_probe full rollouts per decision point, so keep K_probe and the
number of decisions small.

Reports, per K:
  learned_pick  P(success | executing the learned verifier's choice)
  oracle_pick   P(success | executing the highest 16-step-return choice)
  mean_cand     P(success | executing a uniformly random choice)

If oracle_pick sits BELOW mean_cand, the 16-step objective is misspecified and
search amplifies that error regardless of model quality.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from ttc.envs.make import make_env
from ttc.search.base import candidate_seed
from ttc.search.learned import LearnedDynamicsSelector
from ttc.search.oracle import OracleSelector
from scripts.run_sweep import load_dynamics, load_policy, load_value


def rollout_to_end(env, pol, chunk, n_exec, max_steps):
    """Execute chunk[:n_exec], then run the base policy to termination."""
    succ, steps = False, 0
    for act in chunk[:n_exec]:
        obs, _, term, trunc, info = env.step(act)
        steps += 1
        succ = succ or bool(info.get("success", 0.0))
        if term or trunc or succ:
            return succ
    while steps < max_steps:
        act_chunk = pol.sample(obs, n=1, seed=steps)[0]
        for act in act_chunk[:n_exec]:
            obs, _, term, trunc, info = env.step(act)
            steps += 1
            succ = succ or bool(info.get("success", 0.0))
            if term or trunc or succ:
                return succ
    return succ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--k-probe", type=int, default=16)
    p.add_argument("--decisions-per-episode", type=int, default=4)
    p.add_argument("--plan-horizon", type=int, default=4)
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--out", default=None,
                   help="append results as TSV for cross-task aggregation")
    a = p.parse_args()

    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    dyn = load_dynamics(f"experiments/ckpt/dyn_{a.task}.pt")
    val = load_value(f"experiments/ckpt/val_{a.task}.pt")
    learned = LearnedDynamicsSelector(dyn, val, beta=0.0, horizon=a.plan_horizon)
    oracle = OracleSelector()

    ks = [k for k in (2, 4, 8, 16, 32, 64) if k <= a.k_probe]
    res = defaultdict(lambda: defaultdict(list))

    for ep in range(a.n_episodes):
        env = make_env(a.task, seed=ep)
        obs, _ = env.reset(seed=ep)
        step = 0
        for d in range(a.decisions_per_episode):
            pool = pol.sample(obs, n=a.k_probe, seed=candidate_seed(ep, step))
            t_scores = oracle.select(obs, pool, env=env, info={"step": step}).scores
            l_scores = learned.select(obs, pool, info={"step": step}).scores

            # True success for every candidate, from this exact state.
            snap = env.save_state()
            succ = []
            for c in pool:
                succ.append(rollout_to_end(env, pol, c, a.n_exec, a.max_steps))
                env.restore_state(snap)
            succ = np.array(succ, dtype=float)

            for k in ks:
                res[k]["learned"].append(succ[int(np.argmax(l_scores[:k]))])
                res[k]["oracle"].append(succ[int(np.argmax(t_scores[:k]))])
                res[k]["mean"].append(succ[:k].mean())

            chunk = pool[int(np.argmax(l_scores))]
            for act in chunk[:a.n_exec]:
                obs, _, term, trunc, info = env.step(act)
                step += 1
                if term or trunc or info.get("success", 0.0):
                    break
            if term or trunc or info.get("success", 0.0):
                break
        env.close()
        print(f"  episode {ep} done", flush=True)

    n = len(res[ks[0]]["mean"])
    print(f"\n{a.task}  {n} decision points, {a.k_probe} candidates each\n")
    print(f"{'K':>4s} {'learned_pick':>13s} {'oracle_pick':>12s} {'mean_cand':>10s}")
    rows = []
    for k in ks:
        L, O, M = (float(np.mean(res[k][x])) for x in ("learned", "oracle", "mean"))
        print(f"{k:4d} {L:13.3f} {O:12.3f} {M:10.3f}")
        rows.append((k, L, O, M))

    if a.out:
        import os
        new_file = not os.path.exists(a.out)
        with open(a.out, "a") as f:
            if new_file:
                f.write("task\tk\tn\tlearned\toracle\tmean\n")
            for k, L, O, M in rows:
                f.write(f"{a.task}\t{k}\t{n}\t{L:.4f}\t{O:.4f}\t{M:.4f}\n")


if __name__ == "__main__":
    main()
