"""Run trained BC policies closed-loop on the same test scenes as the replays.

Every step: robosuite observation -> 11 features -> policy -> action -> robot. No path, no clock.

Usage:  python scripts/evaluate_policy.py clean_12 [--seeds 0 1 2] [--scenes 50] [--gifs 2] [--tag _time]
        python scripts/evaluate_policy.py clean_12 --ensemble      (average the seeds into one policy)
Output: outputs/policy/bc_<clip>_seed<k>.json (per scene), GIFs, summary (mean +/- std over seeds)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.policy.features import obs_features  # noqa: E402
from src.policy.model import BCPolicy  # noqa: E402
from src.sim.cube_env import CubeToTarget  # noqa: E402
from src.sim.generator import sample_scenes  # noqa: E402
from src.sim.replay import CAMERA, CONTROL_HZ, LIFT_THRESHOLD, load_demo  # noqa: E402

MODELS = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "policy"
MAX_STEPS = 300            # 15 s at 20 Hz; expert episodes average ~186 steps


class Ensemble:
    """Average several policies: mean of the movement outputs, mean of the gripper probabilities."""

    def __init__(self, models):
        self.models = models

    @torch.no_grad()
    def act(self, obs_features):
        x = torch.as_tensor(obs_features, dtype=torch.float32)[None]
        outs = [m(x) for m in self.models]
        moves = [move[0] for move, _ in outs]
        probs = [torch.sigmoid(logit[0]) for _, logit in outs]
        action = torch.zeros(7)
        action[:3] = torch.stack(moves).mean(0)
        action[6] = 1.0 if torch.stack(probs).mean().item() > 0.5 else -1.0
        return action.numpy()


def load_policy(path):
    ckpt = torch.load(path, weights_only=False)
    model = BCPolicy(n_obs=ckpt["n_obs"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def run_policy(model, cube, target, seed=0, save_gif=False, max_steps=MAX_STEPS, nudge=None):
    """Closed-loop rollout. `nudge` = (step, new_cube_xy): teleport the cube at that step (used later)."""
    env = CubeToTarget(robots="Panda", has_renderer=False, has_offscreen_renderer=save_gif,
                       use_camera_obs=save_gif, camera_names=CAMERA, camera_heights=256, camera_widths=256,
                       control_freq=CONTROL_HZ, cube_x_range=(cube[0], cube[0]), cube_y_range=(cube[1], cube[1]),
                       target_xy=(target[0], target[1]), seed=seed)
    obs = env.reset()
    target_pos = env.target_pos.copy()
    frames = []
    for _ in range(10):                                      # cube settles
        obs, _, _, _ = env.step(np.r_[np.zeros(6), -1.0])
    rest_z = max_z = obs["cube_pos"][2]
    for k in range(max_steps):
        if nudge is not None and k == nudge[0]:
            q = env.sim.data.get_joint_qpos(env.cube.joints[0]).copy()
            q[:2] = nudge[1]
            env.sim.data.set_joint_qpos(env.cube.joints[0], q)
            env.sim.forward()
            obs = env._get_observations(force_update=True)
        obs, _, _, _ = env.step(model.act(obs_features(obs, target_pos)))
        max_z = max(max_z, obs["cube_pos"][2])
        if save_gif and k % 2 == 0:
            frames.append(obs[f"{CAMERA}_image"].copy())
    lifted = bool(max_z > rest_z + LIFT_THRESHOLD)
    placed = bool(env._check_success())
    dist = float(np.linalg.norm(obs["cube_pos"][:2] - target_pos[:2]))
    env.close()
    return {"success": lifted and placed, "lifted": lifted, "placed": placed,
            "max_lift_cm": round(float(max_z - rest_z) * 100, 1), "cube_to_target_cm": round(dist * 100, 1)}, frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default="clean_12")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--scenes", type=int, default=50)
    ap.add_argument("--scene-seed", type=int, default=0, help="0 = the same test scenes as the replays")
    ap.add_argument("--gifs", type=int, default=2)
    ap.add_argument("--tag", default="", help="model variant, e.g. _time")
    ap.add_argument("--ensemble", action="store_true", help="average the listed seeds into ONE policy")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    demo = load_demo(args.clip)
    scenes = sample_scenes(demo["cube"], demo["target"], args.scenes, seed=args.scene_seed)
    name = f"{args.clip}{args.tag}"
    rates = []
    if args.ensemble:
        runs = [("ensemble", Ensemble([load_policy(MODELS / f"bc_{name}_seed{k}.pt") for k in args.seeds]))]
    else:
        runs = [(s, load_policy(MODELS / f"bc_{name}_seed{s}.pt")) for s in args.seeds]
    for s, model in runs:
        results, t0 = [], time.time()
        for i, (cube, target) in enumerate(scenes):
            r, frames = run_policy(model, cube, target, seed=i, save_gif=i < args.gifs)
            r["scene"] = i
            results.append(r)
            if frames:
                iio.imwrite(OUT_DIR / f"bc_{name}_seed{s}_scene{i}.gif", frames,
                            duration=1000 * 2 / CONTROL_HZ, loop=0)
        with open(OUT_DIR / f"bc_{name}_seed{s}.json", "w") as f:
            json.dump(results, f, indent=2)
        wins = sum(r["success"] for r in results)
        rates.append(wins / len(results))
        label = "ensemble" if s == "ensemble" else f"seed {s}"
        print(f"  policy {label}: {wins}/{len(results)} successes  lifted {sum(r['lifted'] for r in results)}  "
              f"mean dist to target {np.mean([r['cube_to_target_cm'] for r in results]):.1f} cm  "
              f"({(time.time() - t0) / len(results):.1f} s/scene)")
    rates = np.array(rates) * 100
    print(f"\nBC policy {name}, {len(scenes)} test scenes, {len(rates)} run(s): {rates.mean():.1f}% +/- {rates.std():.1f}%  "
          f"(per run: {', '.join(f'{r:.0f}%' for r in rates)})")
