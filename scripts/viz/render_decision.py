"""Render a real decision point in MuJoCo: K branched candidate rollouts overlaid
on the live scene, colored by true outcome, with the executed one highlighted.

This is not an illustration -- every arc is an actual simulator rollout of an
actual policy sample, scored by actually running it and rewinding. That is only
possible because SnapshotWrapper restores state exactly (mjSTATE_INTEGRATION
plus warmstart reapplication); without it the branches would perturb each other.

Outputs a still PNG of the fan, and optionally an MP4 that holds on the fan and
then executes the chosen candidate.
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


# ---------------------------------------------------------------- scene geoms

def _connector(geom, width, p0, p1):
    """Draw a capsule between two points. The helper was renamed in mujoco 3.x,
    so try the current name first and fall back to the legacy signature."""
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
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, np.float32))
    _connector(g, width, p0, p1)
    g.label = ""
    scene.ngeom += 1


def add_sphere(scene, p, rgba, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]), np.asarray(p, float),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    g.label = ""
    scene.ngeom += 1


def value_color(v, alpha=1.0):
    """Red -> amber -> green. Encodes the TRUE outcome only; belief is drawn as
    a separate violet channel so the two never get confused."""
    stops = np.array([[0.69, 0.23, 0.13],
                      [0.79, 0.64, 0.15],
                      [0.07, 0.47, 0.35]])
    t = float(np.clip(v, 0, 1)) * 2
    i = 0 if t < 1 else 1
    f = t if t < 1 else t - 1
    c = stops[i] + (stops[i + 1] - stops[i]) * f
    return [c[0], c[1], c[2], alpha]


# ---------------------------------------------------------------- ee tracking

def find_tracker(env):
    """Return a callable giving the gripper position in world coordinates.
    Meta-World exposes this differently across versions, so probe in order."""
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
    raise RuntimeError("could not locate a gripper site or body on this env")


# ---------------------------------------------------------------- the sweep

def branch_candidates(env, pol, dyn, val, plan_h, k, seed, n_exec, gamma=0.99):
    """Sample K candidates; roll each in the TRUE simulator recording the gripper
    path and discounted return; rewind between each. Also score with the learned
    verifier so belief and reality can be compared on the same candidates."""
    obs = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else None
    if obs is None:
        raise RuntimeError("cannot read current observation from env")

    track = find_tracker(env)
    pool = pol.sample(obs, n=k, seed=seed)

    learned = LearnedDynamicsSelector(dyn, val, beta=0.0, horizon=plan_h)
    belief = learned.select(obs, pool, info={"step": 0}).scores

    snap = env.save_state()
    paths, truth = [], []
    for c in pool:
        pts, total, disc = [track()], 0.0, 1.0
        for a in c[:n_exec]:
            _, r, term, trunc, info = env.step(a)
            pts.append(track())
            total += disc * float(r)
            if info.get("success", 0.0):
                total += disc * 10.0
            disc *= gamma
            if term or trunc:
                break
        paths.append(np.array(pts))
        truth.append(total)
        env.restore_state(snap)

    truth = np.array(truth)
    norm = ((truth - truth.min()) / (np.ptp(truth) + 1e-9)) if len(truth) > 1 \
        else np.ones_like(truth)
    return pool, paths, truth, norm, belief, snap


def draw_fan(renderer, cam, env, paths, norm, pick, show_pick=True):
    renderer.update_scene(env._data, cam)
    s = renderer.scene
    order = np.argsort(norm)          # draw good paths last so they sit on top
    for idx in order:
        if idx == pick:
            continue
        pts, a = paths[idx], 0.40
        for j in range(len(pts) - 1):
            add_segment(s, pts[j], pts[j + 1], value_color(norm[idx], a), 0.0025)
        add_sphere(s, pts[-1], value_color(norm[idx], a + 0.15), 0.006)

    if show_pick:
        pts = paths[pick]
        for j in range(len(pts) - 1):     # violet halo = what the scorer believes
            add_segment(s, pts[j], pts[j + 1], [0.29, 0.23, 0.77, 0.55], 0.0072)
        for j in range(len(pts) - 1):
            add_segment(s, pts[j], pts[j + 1], value_color(norm[pick], 1.0), 0.0038)
        add_sphere(s, pts[-1], [0.29, 0.23, 0.77, 0.95], 0.014)
        add_sphere(s, pts[-1], value_color(norm[pick], 1.0), 0.009)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="disassemble-v3")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--episode", type=int, default=3)
    p.add_argument("--warmup", type=int, default=48,
                   help="steps to run before branching, so the arm is mid-task")
    p.add_argument("--n-exec", type=int, default=8)
    p.add_argument("--plan-horizon", type=int, default=4)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--azimuth", type=float, default=138)
    p.add_argument("--elevation", type=float, default=-22)
    p.add_argument("--distance", type=float, default=1.5)
    p.add_argument("--video", action="store_true",
                   help="also write an mp4 that holds on the fan then executes")
    p.add_argument("--outdir", default="experiments/figs/viz")
    a = p.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    pol = load_policy(f"experiments/ckpt/bc_{a.task}.pt")
    dyn = load_dynamics(f"experiments/ckpt/dyn_{a.task}.pt")
    val = load_value(f"experiments/ckpt/val_{a.task}.pt")

    env = make_env(a.task, seed=a.episode)
    obs, _ = env.reset(seed=a.episode)
    for t in range(a.warmup):                       # drive to a mid-task state
        chunk = pol.sample(obs, n=1, seed=a.episode * 100 + t)[0]
        for act in chunk[:a.n_exec]:
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
        if term or trunc:
            break

    pool, paths, truth, norm, belief, snap = branch_candidates(
        env, pol, dyn, val, a.plan_horizon, a.k,
        candidate_seed(a.episode, a.warmup), a.n_exec)

    picks = {
        "learned": int(np.argmax(belief)),
        "oracle":  int(np.argmax(truth)),
        "random":  int(np.random.default_rng(a.episode).integers(len(pool))),
    }

    # Offscreen buffer must be sized before the Renderer is built.
    env._model.vis.global_.offwidth = a.width
    env._model.vis.global_.offheight = a.height
    renderer = mujoco.Renderer(env._model, height=a.height, width=a.width,
                               max_geom=40000)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = a.azimuth, a.elevation, a.distance
    # Frame the fan where it originates. nsite is on MjModel, not MjData,
    # and the gripper is a better focal point than the site centroid anyway.
    cam.lookat[:] = paths[0][0] if len(paths) else np.array([0.0, 0.6, 0.15])

    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("pip install imageio imageio-ffmpeg")

    for name, pick in picks.items():
        env.restore_state(snap)
        draw_fan(renderer, cam, env, paths, norm, pick)
        imageio.imwrite(out / f"{a.task}_K{a.k}_{name}.png", renderer.render())
        print(f"wrote {out}/{a.task}_K{a.k}_{name}.png  "
              f"picked #{pick}  true={truth[pick]:.2f}  "
              f"best={truth.max():.2f}  mean={truth.mean():.2f}")

    env.restore_state(snap)
    draw_fan(renderer, cam, env, paths, norm, 0, show_pick=False)
    imageio.imwrite(out / f"{a.task}_K{a.k}_nopick.png", renderer.render())

    if a.video:
        for name, pick in (("learned", picks["learned"]), ("oracle", picks["oracle"])):
            frames = []
            env.restore_state(snap)
            draw_fan(renderer, cam, env, paths, norm, pick)
            hold = renderer.render()
            frames += [hold] * 45                    # hold on the fan
            env.restore_state(snap)
            for act in pool[pick][:a.n_exec]:        # then execute the choice
                env.step(act)
                renderer.update_scene(env._data, cam)
                frames.append(renderer.render())
            frames += [frames[-1]] * 20
            path = out / f"{a.task}_K{a.k}_{name}.mp4"
            imageio.mimwrite(path, frames, fps=24, quality=8)
            print(f"wrote {path}")

    env.close()


if __name__ == "__main__":
    main()
