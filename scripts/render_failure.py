"""Render a GIF of ONE scene from any experiment (e.g. a failure picked with failure_summary.py).

Experiments (same scenes and seeds as the original runs, so the outcome is reproduced):
  raw | object_relative        evaluate_generalisation.py   (--source picks the demo, default clean_12)
  open_loop | replanned | bc   evaluate_nudge.py            (--nudge-step / --nudge-cm as in that run)
  bc_seed                      a single BC seed, no nudge   (--bc-seed k)

Usage:  python scripts/render_failure.py raw 6
        python scripts/render_failure.py bc_seed 2 --bc-seed 0
        python scripts/render_failure.py replanned 3 --nudge-step 70
        python scripts/render_failure.py object_relative 4 --source messy_5
Output: outputs/failures/<experiment>_<source>_scene<N>.gif and the episode's result
"""
import argparse
import sys
from pathlib import Path

import imageio.v3 as iio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_nudge import nudge_step_from_dataset, nudged_positions, run_replay_nudged  # noqa: E402
from evaluate_policy import Ensemble, load_policy, run_policy  # noqa: E402
from src.sim.generator import generate, sample_scenes  # noqa: E402
from src.sim.replay import CONTROL_HZ, load_demo, run_episode  # noqa: E402

MODELS = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "outputs" / "failures"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["raw", "object_relative", "open_loop", "replanned", "bc", "bc_seed"])
    ap.add_argument("scene", type=int)
    ap.add_argument("--source", default="clean_12", help="source demo")
    ap.add_argument("--nudge-step", type=int, default=None)
    ap.add_argument("--nudge-cm", type=float, default=5.0)
    ap.add_argument("--bc-seed", type=int, default=0)
    ap.add_argument("--bc-tag", default="")
    ap.add_argument("--n-scenes", type=int, default=50, help="as in the original run (scene sampling)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    demo = load_demo(args.source)
    scenes = sample_scenes(demo["cube"], demo["target"], args.n_scenes, seed=0)
    cube, target = scenes[args.scene]
    i = args.scene

    if args.experiment in ("raw", "object_relative"):
        path = demo["path"] if args.experiment == "raw" else \
            generate(demo["path"], demo["grip"], demo["cube"], demo["target"], cube, target)
        r, frames = run_episode(path, demo["grip"], cube, target, seed=i, fixes=True, save_gif=True)
        frames = frames[::2]
    elif args.experiment == "bc_seed":
        model = load_policy(MODELS / f"bc_{args.source}{args.bc_tag}_seed{args.bc_seed}.pt")
        r, frames = run_policy(model, cube, target, seed=i, save_gif=True)
    else:
        step = args.nudge_step if args.nudge_step is not None else nudge_step_from_dataset(args.source)
        new_cube = nudged_positions(scenes, args.nudge_cm)[i]
        if args.experiment == "bc":
            model = Ensemble([load_policy(MODELS / f"bc_{args.source}{args.bc_tag}_seed{k}.pt") for k in (0, 1, 2)])
            r, frames = run_policy(model, cube, target, seed=i, save_gif=True, nudge=(step, new_cube[:2]))
        else:
            r, frames = run_replay_nudged(demo, cube, target, new_cube, step, replan=(args.experiment == "replanned"),
                                          seed=i, save_gif=True)
            frames = frames[::2]
    name = f"{args.experiment}_{args.source}_scene{i}" + (f"_seed{args.bc_seed}" if args.experiment == "bc_seed" else "")
    out = OUT_DIR / f"{name}.gif"
    iio.imwrite(out, frames, duration=1000 * 2 / CONTROL_HZ, loop=0)
    print(f"{name}: success {r['success']}  lifted {r['lifted']}  dist to target {r['cube_to_target_cm']} cm -> {out}")
