"""Replay recorded human demos in sim, with the cube and target at their RECORDED positions.

The machinery (table -> sim mapping, resampling to 20 Hz, contact fixes, the controller) lives in
src/sim/replay.py, shared with scripts/evaluate_generalisation.py.

Usage: python scripts/replay_in_sim.py clean_3 [more clips] [--raw] [--trials 5] [--no-gif]
       --raw = pure replay (no contact fixes, no grip pause), the honest baseline
Output: outputs/replay/<clip>_<mode>.json (per trial), outputs/replay/<clip>.gif (trial 0)
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sim.replay import CONTROL_HZ, load_demo, run_episode  # noqa: E402

OUT_DIR = PROJECT_ROOT / "outputs" / "replay"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*", default=["clean_3"])
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--raw", action="store_true", help="pure replay: no contact fixes and no grip pause")
    ap.add_argument("--grip-pause", type=int, default=0, help="steps to hold still while the gripper moves")
    ap.add_argument("--trials", type=int, default=5,
                    help="replays per clip with different seeds (small random robot start pose)")
    args = ap.parse_args()
    mode = "raw" if args.raw else "fixes"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for c in args.clips:
        demo = load_demo(c)
        results = []
        for k in range(args.trials):
            r, frames = run_episode(demo["path"], demo["grip"], demo["cube"], demo["target"], seed=k,
                                    fixes=not args.raw, grip_pause=0 if args.raw else args.grip_pause,
                                    save_gif=(not args.no_gif and k == 0))
            results.append(r)
            if frames:
                iio.imwrite(OUT_DIR / f"{c}.gif", frames[::2], duration=1000 * 2 / CONTROL_HZ, loop=0)
        wins = sum(r["success"] for r in results)
        summary.append((c, wins, len(results), np.mean([r["cube_to_target_cm"] for r in results]),
                        results[0]["path_z_range_cm"][0], results[0]["points_changed"], results[0]["points_total"]))
        with open(OUT_DIR / f"{c}_{mode}.json", "w") as f:
            json.dump({"demo": {"cube": demo["cube"].tolist(), "target": demo["target"].tolist(),
                                "mapping": demo["mapping"]}, "trials": results}, f, indent=2)
        print(f"{c}: {wins}/{len(results)} successes ({mode})  lifts {[r['max_lift_cm'] for r in results]} cm")

    total, trials = sum(s[1] for s in summary), sum(s[2] for s in summary)
    print(f"\n{'clip':10s} {'success':>8s} {'final dist':>10s} {'min z':>8s}  points changed   [{mode}]")
    for c, w, n, dist, mz, ch, tot in summary:
        print(f"{c:10s} {w:>4d}/{n:<3d} {dist:8.1f} cm {mz:6.1f} cm  {ch}/{tot}")
    print(f"TOTAL      {total}/{trials} = {100 * total / trials:.0f}%")
