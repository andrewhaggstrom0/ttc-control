"""Find a decision point where the outcome is genuinely undecided.

Most states are not pivotal: early on nothing has gone wrong yet, and late in a
successful episode any reasonable motion finishes the task, so every future
looks the same. A useful figure needs a state where roughly half the candidate
futures succeed.

Scans (episode, warmup) pairs with a small K and short horizon, and ranks by how
close the success fraction is to 0.5. Cheap enough to sweep dozens of states.
"""

from __future__ import annotations

import argparse

import numpy as np

from ttc.envs.make import make_env
from ttc.search.base import candidate_seed
from scripts.viz.render_futures import find_tracker, get_obs
from scripts.run_sweep import load_policy


def probe(env, pol, k, seed, n_exec, horizon):
    """Success fraction over K futures from the current state."""
    obs0 = get_obs(env)
    pool = pol.sample(obs0, n=k, seed=seed)
    snap = env.save_state()
    wins = 0
    for ci, chunk in enumerate(pool):
        obs, steps = obs0, 0
        success = term = trunc = False
        for a in chunk[:n_exec]:
            obs, _, term, trunc, info = env.step(a)
            steps += 1
            success = success or bool(info.get("success", 0.0))
            if term or trunc or success:
                break
        while steps < horizon and not (success or term or trunc):
            nxt = pol.sample(obs, n=1, seed=seed + steps)[0]
            for a in nxt[:n_exec]:
                obs, _, term, trunc, info = env.step(a)
                steps += 1
                success = success or bool(info.get("success", 0.0))
                if term or trunc or success or steps >= horizon:
                    break
        wins += int(success)
        if steps <= 1:
            # Futures that end immediately mean the state is already
            # terminal; its success fraction is meaningless.
            env.restore_state(snap)
            return -1.0
        env.restore_state(snap)
    return wins / len(pool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="disassemble-v3")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--warmups", type=int, nargs="+",
                   default=[8, 16, 24, 32, 40, 48, 56, 64, 72, 88])
    p.add_argument("--k", type=int, default=8, help="small K keeps the scan cheap")
    p.add_argument("--horizon", type=int, default=140)
    p.add_argument("--n-exec", type=int, default=8)
    a = p.parse_args()

    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    results = []

    for ep in a.episodes:
        env = make_env(a.task, seed=ep, max_steps=1000)
        obs, _ = env.reset(seed=ep)
        step = 0
        term = trunc = False
        for w in sorted(a.warmups):
            while step < w and not (term or trunc):
                chunk = pol.sample(obs, n=1, seed=ep * 100 + step)[0]
                for act in chunk[:a.n_exec]:
                    obs, _, term, trunc, _info = env.step(act)
                    step += 1
                    if _info.get('success', 0.0):
                        term = True   # stop warmup before the task is solved
                    if term or trunc:
                        break
            if term or trunc:
                break
            frac = probe(env, pol, a.k, candidate_seed(ep, w), a.n_exec, a.horizon)
            if frac < 0:
                print(f'  ep {ep:2d}  warmup {w:3d}  ->  already terminal, skipped',
                      flush=True)
                continue
            results.append((abs(frac - 0.5), ep, w, frac))
            print(f"  ep {ep:2d}  warmup {w:3d}  ->  {frac:.2f} of futures succeed",
                  flush=True)
        env.close()

    results.sort()
    print("\nmost pivotal decision points (closest to a 50/50 split):")
    for d, ep, w, frac in results[:6]:
        print(f"  --episode {ep} --warmup {w}   ({frac:.2f} succeed)")
    if results:
        _, ep, w, frac = results[0]
        print(f"\nnext:\n  PYTHONPATH=. python scripts/viz/plot_futures.py \\\n"
              f"    --task {a.task} --episode {ep} --warmup {w} --k 16 --horizon {a.horizon}\n"
              f"  PYTHONPATH=. python scripts/viz/plot_paths.py")


if __name__ == "__main__":
    main()
