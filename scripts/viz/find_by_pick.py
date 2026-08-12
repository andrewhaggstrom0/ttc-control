"""Scan decision points and report what each scorer chose at each one.

find_pivotal.py ranks states by how evenly futures split. This goes further and
records WHICH candidate each scorer picked and whether that future succeeded, so
you can select a state illustrating a specific case:

  learned picks a failure   -- the paper's headline, per decision
  learned picks a success   -- the common case; harm is ~5% per decision and
                               only compounds over the ~30 decisions in an episode
  learned and oracle differ -- the cleanest side-by-side

Both cases are worth showing. A figure with only failures would overstate what
a single decision does.
"""

from __future__ import annotations

import argparse

import numpy as np

from ttc.envs.make import make_env
from ttc.search.base import candidate_seed
from scripts.viz.render_futures import roll_futures, picks_for
from scripts.run_sweep import load_dynamics, load_policy, load_value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="disassemble-v3")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--warmups", type=int, nargs="+",
                   default=[8, 16, 24, 32, 40, 48, 56, 64])
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--horizon", type=int, default=140)
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--plan-horizon", type=int, default=4)
    a = p.parse_args()

    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    dyn = load_dynamics(f"experiments/ckpt/dyn_{a.task}.pt")
    val = load_value(f"experiments/ckpt/val_{a.task}.pt")

    rows = []
    print(f"{'ep':>3s} {'warm':>5s} {'succ/K':>7s} {'learned':>9s} "
          f"{'oracle':>9s} {'random':>9s}")

    for ep in a.episodes:
        env = make_env(a.task, seed=ep, max_steps=1000)
        obs, _ = env.reset(seed=ep)
        step = 0
        term = trunc = False

        for w in sorted(a.warmups):
            while step < a.warmups[-1] and step < w and not (term or trunc):
                chunk = pol.sample(obs, n=1, seed=ep * 100 + step)[0]
                for act in chunk[:a.n_exec]:
                    obs, _, term, trunc, info = env.step(act)
                    step += 1
                    if info.get("success", 0.0):
                        term = True   # never branch from a solved state
                    if term or trunc or step >= w:
                        break
            if term or trunc:
                break

            _, futures, belief, snap = roll_futures(
                env, pol, dyn, val, a.k, candidate_seed(ep, w),
                a.n_exec, a.plan_horizon, a.horizon)
            env.restore_state(snap)

            if max(f["steps"] for f in futures) <= 1:
                print(f"{ep:3d} {w:5d}   already terminal, skipped")
                continue

            succ = np.array([f["success"] for f in futures])
            picks = picks_for(futures, belief, ep)
            tag = lambda n: ("OK " if succ[picks[n]] else "FAIL") + f" #{picks[n]:02d}"
            print(f"{ep:3d} {w:5d} {int(succ.sum()):3d}/{a.k:<3d} "
                  f"{tag('learned'):>9s} {tag('oracle'):>9s} {tag('random'):>9s}")
            rows.append(dict(ep=ep, w=w, frac=succ.mean(),
                             learned_ok=bool(succ[picks["learned"]]),
                             oracle_ok=bool(succ[picks["oracle"]]),
                             differ=picks["learned"] != picks["oracle"]))
        env.close()

    def best(rows, want_learned_ok):
        # Prefer an even split so the figure has both colors, and prefer states
        # where learned and oracle actually chose differently.
        cand = [r for r in rows if r["learned_ok"] == want_learned_ok
                and 0.25 <= r["frac"] <= 0.75]
        if not cand:
            return None
        cand.sort(key=lambda r: (not r["differ"], abs(r["frac"] - 0.5)))
        return cand[0]

    print("\nrecommended states:")
    for label, want in (("learned picks a SUCCESS", True),
                        ("learned picks a FAILURE", False)):
        r = best(rows, want)
        if r is None:
            print(f"  {label}: none found with a mixed outcome")
            continue
        print(f"  {label}:  --episode {r['ep']} --warmup {r['w']}   "
              f"({r['frac']:.2f} of futures succeed"
              f"{', scorers differ' if r['differ'] else ''})")


if __name__ == "__main__":
    main()
