"""Does one demo generalise to new cube / target positions? Raw replay vs object-relative replay.

The same N random scenes (fixed seed) are used for every method:
  raw             the recorded path, unchanged, whatever the new cube/target positions are
  object_relative the path re-attached to the new positions (src/sim/generator.py)
Contact fixes are on for both (the default from the recorded-position experiment).

Usage: python scripts/evaluate_generalisation.py clean_12 [--scenes 50] [--gifs 3]
       python scripts/evaluate_generalisation.py clean_12 --cube-range 0.15 0.12 --target-range 0.15 0.12 \
              --tag _wide --methods object_relative          (wider test: where does it break?)
       python scripts/evaluate_generalisation.py clean_12 --position-noise 0 1 2 3 4 5
              (error budget: the generator gets cube/target positions that are off by N cm, in a
               random direction; the real objects stay where they are)
Output: outputs/eval/<clip><tag>_<method>.json (per scene), GIFs, summary table
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sim.generator import generate, sample_scenes  # noqa: E402
from src.sim.replay import CONTROL_HZ, load_demo, run_episode  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "eval"
METHODS = ["raw", "object_relative"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default="clean_12")
    ap.add_argument("--scenes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="scene seed (same scenes for every method)")
    ap.add_argument("--gifs", type=int, default=3, help="save GIFs for the first N scenes of each method")
    ap.add_argument("--cube-range", type=float, nargs=2, default=[0.08, 0.06],
                    help="max shift of the cube from its recorded position, x and y (m)")
    ap.add_argument("--target-range", type=float, nargs=2, default=[0.08, 0.06],
                    help="max shift of the target from its recorded position, x and y (m)")
    ap.add_argument("--min-sep", type=float, default=0.15, help="minimum cube-target distance (m)")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--tag", default="", help="added to output names, e.g. _wide")
    ap.add_argument("--position-noise", type=float, nargs="+", default=None, metavar="CM",
                    help="error-budget sweep: object-position errors in cm (object-relative only)")
    ap.add_argument("--mark", type=float, nargs="*", default=[],
                    help="measured perception errors (cm) to mark on the error-budget plot")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    demo = load_demo(args.clip)
    scenes = sample_scenes(demo["cube"], demo["target"], args.scenes, seed=args.seed,
                           cube_range=tuple(args.cube_range), target_range=tuple(args.target_range),
                           min_separation=args.min_sep)
    name = f"{args.clip}{args.tag}"
    print(f"{args.clip}: recorded cube {np.round(demo['cube'][:2], 3)}, target {np.round(demo['target'][:2], 3)}; "
          f"{len(scenes)} scenes (seed {args.seed}); cube range +/-{args.cube_range} m, "
          f"target range +/-{args.target_range} m, min separation {args.min_sep} m")

    if args.position_noise is not None:
        rows = []
        for noise_cm in args.position_noise:
            results = []
            for i, (cube, target) in enumerate(scenes):
                rng = np.random.default_rng(1000 + i)          # same error DIRECTIONS at every noise level
                a_c, a_t = rng.uniform(0, 2 * np.pi, 2)
                err = noise_cm / 100
                est_cube = cube + err * np.array([np.cos(a_c), np.sin(a_c), 0.0])
                est_target = target + err * np.array([np.cos(a_t), np.sin(a_t), 0.0])
                path = generate(demo["path"], demo["grip"], demo["cube"], demo["target"], est_cube, est_target)
                r, _ = run_episode(path, demo["grip"], cube, target, seed=i, fixes=True,
                                   est_cube=est_cube, est_target=est_target)
                r.update({"scene": i, "noise_cm": noise_cm, "cube": np.round(cube, 4).tolist(),
                          "target": np.round(target, 4).tolist(),
                          "cube_offset": np.round(cube[:2] - demo["cube"][:2], 4).tolist(),
                          "target_offset": np.round(target[:2] - demo["target"][:2], 4).tolist()})
                results.append(r)
            with open(OUT_DIR / f"{name}_noise{noise_cm:g}cm_object_relative.json", "w") as f:
                json.dump(results, f, indent=2)
            n = len(results)
            row = (noise_cm, sum(r["success"] for r in results), sum(r["lifted"] for r in results), n,
                   float(np.mean([r["cube_to_target_cm"] for r in results])))
            rows.append(row)
            print(f"  position error {noise_cm:4.1f} cm: success {row[1]}/{n}  lifted {row[2]}/{n}  "
                  f"mean dist to target {row[4]:.1f} cm")

        print(f"\n{'error (cm)':>10s} {'success':>10s} {'lifted':>8s}")
        for nc, s_, l_, n, _ in rows:
            print(f"{nc:10.1f} {s_:>5d}/{n:<4d} {l_:>4d}/{n:<3d}")
        x = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(x, [100 * r[1] / r[3] for r in rows], "o-", label="success (lifted and placed)")
        ax.plot(x, [100 * r[2] / r[3] for r in rows], "s--", alpha=0.6, label="cube lifted")
        for m in args.mark:
            ax.axvline(m, color="grey", ls=":", lw=1.5)
            ax.text(m, 5, f" measured\n {m:g} cm", fontsize=8, color="grey")
        ax.set_xlabel("error in the cube and target positions given to the generator (cm)")
        ax.set_ylabel("success rate (%)")
        ax.set_ylim(-3, 103)
        ax.set_title(f"{args.clip}: how much perception error can object-relative replay tolerate?")
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = OUT_DIR / f"{name}_error_budget.png"
        plt.savefig(out, dpi=120)
        print(f"\nError budget plot: {out}")
        sys.exit(0)

    summary = {}
    for method in args.methods:
        results, t0 = [], time.time()
        for i, (cube, target) in enumerate(scenes):
            if method == "raw":
                path = demo["path"]
            else:
                path = generate(demo["path"], demo["grip"], demo["cube"], demo["target"], cube, target)
            r, frames = run_episode(path, demo["grip"], cube, target, seed=i, fixes=True,
                                    save_gif=i < args.gifs)
            r.update({"scene": i, "cube": np.round(cube, 4).tolist(), "target": np.round(target, 4).tolist(),
                      "cube_shift_cm": round(float(np.linalg.norm(cube[:2] - demo["cube"][:2])) * 100, 1),
                      "target_shift_cm": round(float(np.linalg.norm(target[:2] - demo["target"][:2])) * 100, 1),
                      "cube_offset": np.round(cube[:2] - demo["cube"][:2], 4).tolist(),
                      "target_offset": np.round(target[:2] - demo["target"][:2], 4).tolist(),
                      "cube_range": args.cube_range, "target_range": args.target_range})
            results.append(r)
            if frames:
                iio.imwrite(OUT_DIR / f"{name}_{method}_scene{i}.gif", frames[::2],
                            duration=1000 * 2 / CONTROL_HZ, loop=0)
            done = i + 1
            if done % 10 == 0 or done == len(scenes):
                rate = np.mean([x["success"] for x in results]) * 100
                print(f"  {method:16s} {done:3d}/{len(scenes)}  success so far {rate:5.1f}%  "
                      f"({(time.time() - t0) / done:.1f} s/scene)")
        with open(OUT_DIR / f"{name}_{method}.json", "w") as f:
            json.dump(results, f, indent=2)
        summary[method] = results

    print(f"\n{'method':16s} {'success':>10s} {'lifted':>8s} {'mean dist to target':>20s}")
    for method, res in summary.items():
        n = len(res)
        print(f"{method:16s} {sum(r['success'] for r in res):>4d}/{n:<4d}  {sum(r['lifted'] for r in res):>4d}/{n:<3d}"
              f" {np.mean([r['cube_to_target_cm'] for r in res]):14.1f} cm")
    print(f"\nPer-scene results: {OUT_DIR}/{name}_<method>.json   GIFs: {OUT_DIR}/")
