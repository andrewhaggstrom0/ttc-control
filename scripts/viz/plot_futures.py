"""Roll K futures from one decision point, save them, and plot them in 2D.

The 3D MuJoCo overlay is unreadable: trails are thin, the scene is cluttered,
and the paths overlap in depth. Projected into 2D they are unambiguous.

Data collection and plotting are separate so the expensive part (rolling K
trajectories through the simulator) runs once and can be re-plotted freely.

Panels:
  A  top-down XY -- where the gripper actually goes
  B  height over time -- lift/place structure the top view hides
  C  divergence -- pairwise spread vs step, showing candidates are identical at
     the decision point and separate only later. This is why the moment of
     choice looks like nothing in the simulator.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ttc.envs.make import make_env
from ttc.search.base import candidate_seed
from scripts.viz.render_futures import find_tracker, get_obs, roll_futures, picks_for
from scripts.run_sweep import load_dynamics, load_policy, load_value

SUCCESS = "#12795A"
FAIL    = "#B03A22"
PICK    = "#4A3AC4"


def collect(a):
    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    dyn = load_dynamics(f"experiments/ckpt/dyn_{a.task}.pt")
    val = load_value(f"experiments/ckpt/val_{a.task}.pt")

    env = make_env(a.task, seed=a.episode, max_steps=1000)
    obs, _ = env.reset(seed=a.episode)
    term = trunc = False
    step = 0
    while step < a.warmup and not (term or trunc):
        chunk = pol.sample(obs, n=1, seed=a.episode * 100 + step)[0]
        for act in chunk[:a.n_exec]:
            obs, _, term, trunc, _info = env.step(act)
            step += 1
            if _info.get("success", 0.0):
                term = True   # never branch from an already-solved state
            if term or trunc or step >= a.warmup:
                break

    print(f"rolling {a.k} futures, up to {a.horizon} steps each...")
    pool, futures, belief, _ = roll_futures(
        env, pol, dyn, val, a.k, candidate_seed(a.episode, a.warmup),
        a.n_exec, a.plan_horizon, a.horizon)
    env.close()

    picks = picks_for(futures, belief, a.episode)
    out = Path(a.data)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        paths=np.array([f["path"] for f in futures], dtype=object),
        success=np.array([f["success"] for f in futures]),
        belief=belief,
        picks=np.array([picks["learned"], picks["oracle"], picks["random"]]),
        task=a.task, n_exec=a.n_exec,
    )
    print(f"saved {out}")
    return out


def plot(a):
    d = np.load(a.data, allow_pickle=True)
    paths = [np.asarray(p, dtype=float) for p in d["paths"]]
    succ = d["success"]
    picks = dict(zip(("learned", "oracle", "random"), d["picks"]))
    n_exec = int(d["n_exec"])
    task = str(d["task"])
    K = len(paths)

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.26)
    axA, axB, axC = (fig.add_subplot(gs[i]) for i in range(3))

    order = sorted(range(K), key=lambda i: succ[i])   # winners drawn on top
    pick = picks["learned"]

    # -- A: top-down XY -----------------------------------------------------
    for i in order:
        p = paths[i]
        c = SUCCESS if succ[i] else FAIL
        is_pick = i == pick
        if is_pick:
            axA.plot(p[:, 0], p[:, 1], color=PICK, lw=5.5, alpha=.35,
                     solid_capstyle="round", zorder=4)
        axA.plot(p[:, 0], p[:, 1], color=c, lw=2.4 if is_pick else 1.1,
                 alpha=1.0 if is_pick else .55, zorder=5 if is_pick else 2)
        axA.plot(p[-1, 0], p[-1, 1], "o", color=c, ms=7 if is_pick else 3.4,
                 mec="white", mew=.8, zorder=6)

    start = paths[0][0]
    axA.plot(start[0], start[1], "o", color="#141A21", ms=9, zorder=7)
    axA.annotate("decision point", (start[0], start[1]),
                 xytext=(14, 14), textcoords="offset points", fontsize=9,
                 arrowprops=dict(arrowstyle="-", lw=.8, color="#5A6672"))
    axA.set_xlabel("x (m)"); axA.set_ylabel("y (m)")
    axA.set_title(f"Where each choice leads  ({K} futures)", fontsize=11, loc="left")
    axA.set_aspect("equal", adjustable="datalim")
    axA.grid(alpha=.2)

    # -- B: height over time ------------------------------------------------
    for i in order:
        p = paths[i]
        c = SUCCESS if succ[i] else FAIL
        is_pick = i == pick
        axB.plot(np.arange(len(p)), p[:, 2], color=c,
                 lw=2.4 if is_pick else 1.1, alpha=1.0 if is_pick else .5,
                 zorder=5 if is_pick else 2)
    axB.axvspan(0, n_exec, color=PICK, alpha=.10, zorder=1)
    axB.text(n_exec + 2, axB.get_ylim()[1], " the chosen action chunk\n (8 steps)",
             va="top", fontsize=8.5, color="#5A6672")
    axB.set_xlabel("step after decision"); axB.set_ylabel("gripper height (m)")
    axB.set_title("Lift and place structure", fontsize=11, loc="left")
    axB.grid(alpha=.2)

    # -- C: divergence ------------------------------------------------------
    T = min(len(p) for p in paths)
    stack = np.stack([p[:T] for p in paths]).astype(float)
    spread = np.linalg.norm(stack - stack.mean(0), axis=2).mean(0) * 100
    axC.plot(np.arange(T), spread, color="#141A21", lw=2.2)
    axC.fill_between(np.arange(T), 0, spread, color="#141A21", alpha=.07)
    axC.axvspan(0, n_exec, color=PICK, alpha=.12)
    axC.set_xlabel("step after decision")
    axC.set_ylabel("mean spread between futures (cm)")
    axC.set_title("Identical at the moment of choice", fontsize=11, loc="left")
    axC.grid(alpha=.2)
    axC.annotate(f"{spread[min(n_exec, T-1)]:.1f} cm apart\nwhen the robot commits",
                 xy=(n_exec, spread[min(n_exec, T - 1)]),
                 xytext=(T * .30, max(spread) * .55), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=.9, color="#5A6672"))
    axC.annotate(f"{spread[-1]:.0f} cm apart\nby the end",
                 xy=(T - 1, spread[-1]),
                 xytext=(T * .40, max(spread) * .88), fontsize=9,
                 arrowprops=dict(arrowstyle="->", lw=.9, color="#5A6672"))

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=SUCCESS, lw=2.4, label="future succeeds"),
        Line2D([], [], color=FAIL, lw=2.4, label="future fails"),
        Line2D([], [], color=PICK, lw=5.5, alpha=.45, label="chosen by learned verifier"),
    ], loc="lower center", ncol=3, frameon=False, fontsize=9.5,
        bbox_to_anchor=(.5, -.02))

    n_ok = int(succ.sum())
    fig.suptitle(
        f"{task}  \u2014  {n_ok} of {K} futures succeed; "
        f"the verifier chose one that "
        f"{'succeeds' if succ[pick] else 'fails'}",
        fontsize=12.5, x=.5, y=.99)
    fig.tight_layout(rect=[0, .05, 1, .95])

    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=170)
    print(f"wrote {outp}")

    for name, i in picks.items():
        print(f"  {name:8s} -> candidate {i:2d}  "
              f"{'SUCCESS' if succ[i] else 'fails'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="disassemble-v3")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--episode", type=int, default=3)
    p.add_argument("--warmup", type=int, default=48)
    p.add_argument("--horizon", type=int, default=140)
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--plan-horizon", type=int, default=4)
    p.add_argument("--data", default="experiments/figs/viz/futures.npz")
    p.add_argument("--out", default="experiments/figs/viz/futures_2d.png")
    p.add_argument("--replot", action="store_true",
                   help="skip rolling; plot from an existing npz")
    a = p.parse_args()

    if not a.replot:
        collect(a)
    plot(a)


if __name__ == "__main__":
    main()
