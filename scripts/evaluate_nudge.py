"""The nudge experiment: the cube is moved partway through the approach, before the grasp.

Same test scenes as before (seed 0). In each scene the cube jumps NUDGE_CM in a random horizontal
direction at the same control step for every method:
  open_loop   object-relative path generated for the ORIGINAL cube position, executed unchanged
  replanned   the same, but at the nudge the rest of the path is regenerated for the NEW cube position
              (hand-written re-planning: it is told that, and when, the cube moved)
  bc          the behaviour-cloning ensemble (closed loop: it only ever sees the current state)

Usage:  python scripts/evaluate_nudge.py clean_12 [--nudge-cm 5] [--scenes 50] [--bc-seeds 0 1 2] [--gifs 2]
Output: outputs/nudge/<clip>_<method>.json, GIFs, summary table
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_policy import Ensemble, load_policy, run_policy  # noqa: E402
from src.sim.cube_env import CubeToTarget  # noqa: E402
from src.sim.generator import generate, sample_scenes  # noqa: E402
from src.sim.replay import (CAMERA, CONTROL_HZ, LIFT_THRESHOLD, PREROLL_TOL, SETTLE_STEPS,  # noqa: E402
                            contact_fixes, load_demo, move_action)

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "nudge"
MIN_TARGET_GAP = 0.10        # the nudged cube must stay at least 10 cm from the target


def nudge_step_from_dataset(clip):
    """Halfway through the approach: half the median number of steps before the expert's first 'close'."""
    
    path = PROCESSED / f"dataset_{clip}.npz"
    if not path.exists():
        return 40
    d = np.load(path)
    firsts = [np.argmax(d["act_grip"][d["episode"] == e] > 0.5) for e in np.unique(d["episode"])]
    return int(np.median(firsts) / 2)


def nudged_positions(scenes, nudge_cm):
    """For each scene, the cube's new position: nudge_cm in a random direction (fixed per scene)."""

    out = []
    for i, (cube, target) in enumerate(scenes):
        rng = np.random.default_rng(2000 + i)
        for _ in range(100):
            a = rng.uniform(0, 2 * np.pi)
            new = cube.copy()
            new[:2] += nudge_cm / 100 * np.array([np.cos(a), np.sin(a)])
            if np.linalg.norm(new[:2] - target[:2]) >= MIN_TARGET_GAP:
                break
        out.append(new)
    return out


def teleport_cube(env, xy):
    j = env.cube.joints[0]
    q = env.sim.data.get_joint_qpos(j).copy()
    q[:2] = xy
    env.sim.data.set_joint_qpos(j, q)
    env.sim.data.set_joint_qvel(j, np.zeros(6))
    env.sim.forward()
    return env._get_observations(force_update=True)


def run_replay_nudged(demo, cube, target, new_cube, nudge_step, replan, seed, save_gif=False):
    """Object-relative replay with the cube teleported at control step `nudge_step`.
    replan=True: at the nudge, the rest of the path is regenerated for the new cube position."""

    grip = demo["grip"]
    path = generate(demo["path"], grip, demo["cube"], demo["target"], cube, target)
    path, _ = contact_fixes(path, grip, cube[:2], target[:2])
    env = CubeToTarget(robots="Panda", has_renderer=False, has_offscreen_renderer=save_gif, use_camera_obs=save_gif,
                       camera_names=CAMERA, camera_heights=256, camera_widths=256, control_freq=CONTROL_HZ,
                       cube_x_range=(cube[0], cube[0]), cube_y_range=(cube[1], cube[1]),
                       target_xy=(target[0], target[1]), seed=seed)
    obs = env.reset()
    frames, step_count, nudged = [], 0, False

    def step(action):
        nonlocal obs, step_count, nudged, path
        if step_count == nudge_step and not nudged:
            obs = teleport_cube(env, new_cube[:2])
            nudged = True
            if replan:                                   # regenerate for the new cube; same time index
                new_path = generate(demo["path"], grip, demo["cube"], demo["target"], new_cube, target)
                path, _ = contact_fixes(new_path, grip, new_cube[:2], target[:2])
            action = move_action(obs["robot0_eef_pos"], current_target(), current_grip())
        obs, _, _, _ = env.step(action)
        step_count += 1
        if save_gif and step_count % 2 == 0:
            frames.append(obs[f"{CAMERA}_image"].copy())

    k_path = {"k": None}
    current_target = lambda: path[0] if k_path["k"] is None else path[k_path["k"]]   # noqa: E731
    current_grip = lambda: False if k_path["k"] is None else bool(grip[k_path["k"]])  # noqa: E731

    for _ in range(10):                                 # cube settles (not counted)
        obs, _, _, _ = env.step(np.r_[np.zeros(6), -1.0])
    rest_z = max_z = obs["cube_pos"][2]
    for _ in range(200):                                # pre-roll to the start of the path
        if np.linalg.norm(path[0] - obs["robot0_eef_pos"]) < PREROLL_TOL:
            break
        step(move_action(obs["robot0_eef_pos"], path[0], False))
    for k in range(len(path)):
        k_path["k"] = k
        step(move_action(obs["robot0_eef_pos"], path[k], grip[k]))
        max_z = max(max_z, obs["cube_pos"][2])
    for _ in range(SETTLE_STEPS):
        step(move_action(obs["robot0_eef_pos"], obs["robot0_eef_pos"], False))
    lifted = bool(max_z > rest_z + LIFT_THRESHOLD)
    placed = bool(env._check_success())
    dist = float(np.linalg.norm(obs["cube_pos"][:2] - np.asarray(target)[:2]))
    env.close()
    return {"success": lifted and placed, "lifted": lifted, "placed": placed, "nudged": nudged,
            "max_lift_cm": round(float(max_z - rest_z) * 100, 1), "cube_to_target_cm": round(dist * 100, 1)}, frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default="clean_12")
    ap.add_argument("--nudge-cm", type=float, default=5.0)
    ap.add_argument("--nudge-step", type=int, default=None, help="default: halfway through the expert's approach")
    ap.add_argument("--scenes", type=int, default=50)
    ap.add_argument("--bc-seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds averaged into the BC ensemble")
    ap.add_argument("--bc-tag", default="", help="BC model variant, e.g. _time")
    ap.add_argument("--methods", nargs="+", default=["open_loop", "replanned", "bc"],
                    choices=["open_loop", "replanned", "bc"])
    ap.add_argument("--gifs", type=int, default=2)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    demo = load_demo(args.clip)
    scenes = sample_scenes(demo["cube"], demo["target"], args.scenes, seed=0)
    new_cubes = nudged_positions(scenes, args.nudge_cm)
    nudge_step = args.nudge_step if args.nudge_step is not None else nudge_step_from_dataset(args.clip)
    print(f"{args.clip}: {len(scenes)} scenes; cube nudged {args.nudge_cm} cm at control step {nudge_step} "
          f"({nudge_step / CONTROL_HZ:.1f} s, about halfway through the approach)")

    bc = None
    if "bc" in args.methods:
        bc = Ensemble([load_policy(MODELS / f"bc_{args.clip}{args.bc_tag}_seed{s}.pt") for s in args.bc_seeds])

    summary = {}
    for method in args.methods:
        results = []
        for i, ((cube, target), new_cube) in enumerate(zip(scenes, new_cubes)):
            gif = i < args.gifs
            if method == "bc":
                r, frames = run_policy(bc, cube, target, seed=i, save_gif=gif, nudge=(nudge_step, new_cube[:2]))
            else:
                r, frames = run_replay_nudged(demo, cube, target, new_cube, nudge_step,
                                              replan=(method == "replanned"), seed=i, save_gif=gif)
            r.update({"scene": i, "cube": np.round(cube, 4).tolist(), "nudged_cube": np.round(new_cube, 4).tolist()})
            results.append(r)
            if frames:
                iio.imwrite(OUT_DIR / f"{args.clip}_{method}_scene{i}.gif", frames[::2] if method != "bc" else frames,
                            duration=1000 * 2 / CONTROL_HZ, loop=0)
        with open(OUT_DIR / f"{args.clip}_{method}.json", "w") as f:
            json.dump(results, f, indent=2)
        summary[method] = results
        n = len(results)
        print(f"  {method:10s} {sum(r['success'] for r in results):3d}/{n}  "
              f"lifted {sum(r['lifted'] for r in results)}/{n}  "
              f"mean dist to target {np.mean([r['cube_to_target_cm'] for r in results]):.1f} cm")

    print(f"\n{'method':12s} {'success':>10s}   (cube nudged {args.nudge_cm} cm at step {nudge_step})")
    for m, res in summary.items():
        print(f"{m:12s} {sum(r['success'] for r in res):>4d}/{len(res):<4d} = {100 * np.mean([r['success'] for r in res]):.0f}%")
    print(f"\nPer-scene results and GIFs: {OUT_DIR}/")
