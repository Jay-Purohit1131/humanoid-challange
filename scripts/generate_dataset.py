"""Generate a behaviour-cloning dataset from ONE recorded human demo.

For N random scenes, the demo is re-attached to the new cube/target positions (object-relative
generation) and executed with the tracking controller plus small action noise (DART). Every step's
observation and the expert's clean action are logged. Only successful rollouts are kept, and the
generator's own success rate is reported.

Usage:  python scripts/generate_dataset.py clean_12 [--rollouts 300] [--noise 0.1]
Output: data/processed/dataset_<clip>.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.policy.collect import collect_episode  # noqa: E402
from src.policy.features import FEATURE_NAMES  # noqa: E402
from src.sim.generator import generate, sample_scenes  # noqa: E402
from src.sim.replay import load_demo  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default="clean_12")
    ap.add_argument("--rollouts", type=int, default=300, help="scenes to attempt")
    ap.add_argument("--noise", type=float, default=0.1, help="action noise std (action units; 0.1 = 5 mm)")
    ap.add_argument("--cube-range", type=float, nargs=2, default=[0.15, 0.12])
    ap.add_argument("--target-range", type=float, nargs=2, default=[0.15, 0.12])
    ap.add_argument("--seed", type=int, default=1, help="scene seed (different from the test scenes' seed 0)")
    ap.add_argument("--time-contacts", action="store_true",
                    help="old expert: switch the gripper at a fixed step instead of at the contact point")
    ap.add_argument("--tag", default="", help="added to the dataset name, e.g. _time")
    args = ap.parse_args()

    demo = load_demo(args.clip)
    scenes = sample_scenes(demo["cube"], demo["target"], args.rollouts, seed=args.seed,
                           cube_range=tuple(args.cube_range), target_range=tuple(args.target_range))
    keep, attempted, fallbacks, t0 = [], 0, 0, time.time()
    for i, (cube, target) in enumerate(scenes):
        path = generate(demo["path"], demo["grip"], demo["cube"], demo["target"], cube, target)
        log, ok, info = collect_episode(path, demo["grip"], cube, target, seed=10_000 + i, action_noise=args.noise,
                                        state_contacts=not args.time_contacts)
        attempted += 1
        fallbacks += info["fallbacks"]
        if ok:
            keep.append((log, cube, info["target"]))
        if (i + 1) % 25 == 0 or i + 1 == len(scenes):
            print(f"  {i + 1:4d}/{len(scenes)}  kept {len(keep)}  ({(time.time() - t0) / (i + 1):.1f} s/rollout)")

    ep_ids = np.concatenate([np.full(len(ep_log["obs"]), e) for e, (ep_log, _, _) in enumerate(keep)])
    data = {k: np.concatenate([ep_log[k] for ep_log, _, _ in keep]) for k in keep[0][0]}
    out = PROCESSED / f"dataset_{args.clip}{args.tag}.npz"
    meta = {"source_demo": args.clip, "attempted": attempted, "kept": len(keep),
            "generator_success_rate": round(len(keep) / attempted, 3), "action_noise": args.noise,
            "cube_range": args.cube_range, "target_range": args.target_range, "scene_seed": args.seed,
            "contacts": "time" if args.time_contacts else "state", "reach_fallbacks": fallbacks,
            "features": FEATURE_NAMES}
    np.savez_compressed(out, episode=ep_ids, cube_start=np.array([c for _, c, _ in keep]),
                        target=np.array([t for _, _, t in keep]), meta=json.dumps(meta), **data)
    print(f"\nGenerator success rate (with action noise {args.noise}, "
          f"{'time' if args.time_contacts else 'state'}-based contacts): {len(keep)}/{attempted} "
          f"= {100 * len(keep) / attempted:.1f}%   (contact points not reached in time: {fallbacks})")
    print(f"Dataset: {len(ep_ids)} steps from {len(keep)} episodes "
          f"(mean {len(ep_ids) / max(1, len(keep)):.0f} steps/episode) -> {out}")
    print(f"  obs {data['obs'].shape}  act_move {data['act_move'].shape}  act_grip {data['act_grip'].shape}  "
          f"gripper closed in {100 * data['act_grip'].mean():.0f}% of steps")
