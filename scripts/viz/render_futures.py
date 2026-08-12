"""Visualize the futures a robot chooses between at a single decision point.

The naive version -- drawing the 8-step action chunks -- fails because
candidates differ by millimeters at the moment of choice. What separates them is
where they LEAD. So each candidate is executed and then followed under the base
policy to episode end, and the full gripper trajectory is drawn, colored by
whether that future succeeded.

Two outputs:
  --mode trails  a still frame with every future drawn as a path
  --mode ghosts  an mp4 where all K futures play at once as translucent arms,
                 with the executed one solid
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from ttc.envs.make import make_env
from ttc.search.base import candidate_seed
from ttc.search.learned import LearnedDynamicsSelector
from scripts.run_sweep import load_dynamics, load_policy, load_value


# ------------------------------------------------------------------ scene geoms

def _connector(geom, width, p0, p1):
    try:
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(p0, float), np.asarray(p1, float))
    except AttributeError:
        mujoco.mjv_makeConnector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                                 p0[0], p0[1], p0[2], p1[0], p1[1], p1[2])


def add_segment(scene, p0, p1, rgba, width):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    _connector(g, width, p0, p1)
    g.label = ""
    scene.ngeom += 1


def add_sphere(scene, p, rgba, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(p, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    g.label = ""
    scene.ngeom += 1


SUCCESS_RGB = (0.10, 0.72, 0.42)
FAIL_RGB    = (0.85, 0.26, 0.16)
PICK_RGB    = (0.42, 0.30, 0.95)


def outcome_color(success, alpha):
    c = SUCCESS_RGB if success else FAIL_RGB
    return [c[0], c[1], c[2], alpha]


# ------------------------------------------------------------------ ee tracking

def find_tracker(env):
    u = env.unwrapped
    if hasattr(u, "tcp_center"):
        try:
            np.asarray(u.tcp_center, float).reshape(3)
            return lambda: np.asarray(u.tcp_center, float).copy()
        except Exception:
            pass
    if hasattr(u, "get_body_com"):
        for name in ("hand", "rightclaw", "leftpad", "gripper"):
            try:
                np.asarray(u.get_body_com(name), float).reshape(3)
                return lambda n=name: np.asarray(u.get_body_com(n), float).copy()
            except Exception:
                continue
    model, data = env._model, env._data
    for name in ("endEffector", "grip_site", "hand", "tcp"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            return lambda s=sid: data.site_xpos[s].copy()
    raise RuntimeError("could not locate a gripper site or body")


def get_obs(env):
    u = env.unwrapped
    if hasattr(u, "_get_obs"):
        return u._get_obs()
    raise RuntimeError("cannot read observation")


# ------------------------------------------------------------------ the futures

def roll_futures(env, pol, dyn, val, k, seed, n_exec, plan_h, horizon):
    """For each of K candidates: execute the chunk, then continue under the base
    policy for `horizon` steps. Record the full gripper path and the outcome."""
    obs0 = get_obs(env)
    track = find_tracker(env)
    pool = pol.sample(obs0, n=k, seed=seed)

    learned = LearnedDynamicsSelector(dyn, val, beta=0.0, horizon=plan_h)
    belief = learned.select(obs0, pool, info={"step": 0}).scores

    snap = env.save_state()
    futures = []
    for ci, chunk in enumerate(pool):
        pts, actions, obs = [track()], [], obs0
        success, steps = False, 0

        for a in chunk[:n_exec]:                       # the candidate itself
            obs, _, term, trunc, info = env.step(a)
            pts.append(track()); actions.append(np.array(a)); steps += 1
            success = success or bool(info.get("success", 0.0))
            if term or trunc or success:
                break

        while steps < horizon and not (success or term or trunc):
            nxt = pol.sample(obs, n=1, seed=seed + steps)[0]
            for a in nxt[:n_exec]:                     # then the base policy
                obs, _, term, trunc, info = env.step(a)
                pts.append(track()); actions.append(np.array(a)); steps += 1
                success = success or bool(info.get("success", 0.0))
                if term or trunc or success or steps >= horizon:
                    break

        futures.append(dict(path=np.array(pts), actions=np.array(actions),
                            success=success, steps=steps))
        env.restore_state(snap)
        print(f"  candidate {ci:2d}: {steps:3d} steps, "
              f"{'SUCCESS' if success else 'failed'}", flush=True)

    return pool, futures, belief, snap


def picks_for(futures, belief, seed):
    truth = np.array([f["success"] for f in futures], float)
    return {
        "learned": int(np.argmax(belief)),
        "oracle":  int(np.argmax(truth - 0.001 * np.arange(len(truth)))),
        "random":  int(np.random.default_rng(seed).integers(len(futures))),
    }


def draw_trails(renderer, cam, env, futures, pick, stride=3):
    renderer.update_scene(env._data, cam)
    s = renderer.scene
    order = sorted(range(len(futures)), key=lambda i: (i == pick, futures[i]["success"]))
    for i in order:
        f = futures[i]
        p = f["path"][::stride]
        is_pick = i == pick
        if is_pick:
            for j in range(len(p) - 1):
                add_segment(s, p[j], p[j+1], [*PICK_RGB, 0.45], 0.0085)
        w = 0.0042 if is_pick else 0.0016
        a = 1.0 if is_pick else 0.34
        for j in range(len(p) - 1):
            add_segment(s, p[j], p[j+1], outcome_color(f["success"], a), w)
        add_sphere(s, f["path"][-1], outcome_color(f["success"], min(1.0, a + 0.3)),
                   0.013 if is_pick else 0.007)
    add_sphere(s, futures[0]["path"][0], [0.08, 0.10, 0.13, 1.0], 0.011)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="disassemble-v3")
    p.add_argument("--k", type=int, default=12,
                   help="keep low: 32 long trajectories is spaghetti")
    p.add_argument("--episode", type=int, default=3)
    p.add_argument("--warmup", type=int, default=48)
    p.add_argument("--horizon", type=int, default=120,
                   help="steps to follow each future; this is what makes them diverge")
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--plan-horizon", type=int, default=4)
    p.add_argument("--mode", choices=["trails", "ghosts", "both"], default="both")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--azimuth", type=float, default=145)
    p.add_argument("--elevation", type=float, default=-24)
    p.add_argument("--distance", type=float, default=1.05)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--outdir", default="experiments/figs/viz")
    a = p.parse_args()

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("pip install imageio imageio-ffmpeg")

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
    pool, futures, belief, snap = roll_futures(
        env, pol, dyn, val, a.k, candidate_seed(a.episode, a.warmup),
        a.n_exec, a.plan_horizon, a.horizon)

    picks = picks_for(futures, belief, a.episode)
    nsucc = sum(f["success"] for f in futures)
    print(f"\n{nsucc}/{a.k} futures succeed")
    for name, i in picks.items():
        print(f"  {name:8s} -> candidate {i:2d}  "
              f"{'SUCCESS' if futures[i]['success'] else 'FAILS'}")
    if len(set(picks.values())) == 1:
        print("\n[!] all scorers picked the same candidate -- try another "
              "--warmup or --episode to find a decision that actually differs")

    env._model.vis.global_.offwidth = a.width
    env._model.vis.global_.offheight = a.height
    renderer = mujoco.Renderer(env._model, height=a.height, width=a.width,
                               max_geom=60000)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = a.azimuth, a.elevation, a.distance
    allpts = np.concatenate([f["path"] for f in futures])
    cam.lookat[:] = (allpts.min(0) + allpts.max(0)) / 2   # frame all the futures

    if a.mode in ("trails", "both"):
        for name, pick in picks.items():
            env.restore_state(snap)
            draw_trails(renderer, cam, env, futures, pick)
            fp = out / f"{a.task}_futures_K{a.k}_{name}.png"
            imageio.imwrite(fp, renderer.render())
            print(f"wrote {fp}")
        env.restore_state(snap)
        draw_trails(renderer, cam, env, futures, -1)
        imageio.imwrite(out / f"{a.task}_futures_K{a.k}_all.png", renderer.render())

    if a.mode in ("ghosts", "both"):
        T = max(f["steps"] for f in futures)
        acc = np.zeros((T, a.height, a.width, 3), np.float32)
        wsum = np.zeros(T, np.float32)
        pick = picks["learned"]

        for ci, f in enumerate(futures):
            env.restore_state(snap)
            w = 3.0 if ci == pick else 1.0        # executed future renders solid
            last = None
            for t in range(T):
                if t < len(f["actions"]):
                    env.step(f["actions"][t])
                renderer.update_scene(env._data, cam)
                s = renderer.scene
                trail = f["path"][:max(2, t + 2):3]
                for j in range(len(trail) - 1):
                    add_segment(s, trail[j], trail[j+1],
                                outcome_color(f["success"], 0.9 if ci == pick else 0.5),
                                0.004 if ci == pick else 0.0018)
                last = renderer.render().astype(np.float32)
                acc[t] += last * w
                wsum[t] += w
            print(f"  rendered ghost {ci+1}/{len(futures)}", flush=True)

        frames = [np.clip(acc[t] / max(wsum[t], 1e-6), 0, 255).astype(np.uint8)
                  for t in range(T)]
        frames = [frames[0]] * 12 + frames + [frames[-1]] * 18
        fp = out / f"{a.task}_ghosts_K{a.k}.mp4"
        imageio.mimwrite(fp, frames, fps=a.fps, quality=8)
        print(f"wrote {fp}")

    env.close()


if __name__ == "__main__":
    main()
