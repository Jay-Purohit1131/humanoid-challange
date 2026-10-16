"""Every failure across every experiment, by type, with scene numbers (to pick failure GIFs).

Reads the per-scene result files in outputs/{replay,eval,policy,nudge}/ and classifies each failure:
  never lifted         the gripper never got the cube off the table
  lifted, missed       the cube was lifted but not placed inside the target
Usage: python scripts/failure_summary.py
"""
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"


def load(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):                      # replay_in_sim.py: {"demo": ..., "trials": [...]}
        data = data.get("trials", [])
    return data if isinstance(data, list) and data and isinstance(data[0], dict) else None


def main():
    rows = []
    for folder in ["replay", "eval", "policy", "nudge"]:
        for f in sorted(glob.glob(str(OUT / folder / "*.json"))):
            res = load(f)
            if res is None or "success" not in res[0]:
                continue
            key = "scene" if "scene" in res[0] else "seed"
            never = [r.get(key, i) for i, r in enumerate(res) if not r["success"] and not r.get("lifted", False)]
            missed = [r.get(key, i) for i, r in enumerate(res) if not r["success"] and r.get("lifted", False)]
            rows.append((folder, Path(f).stem, len(res), sum(r["success"] for r in res), never, missed))

    print(f"{'folder':7s} {'experiment':40s} {'success':>9s}  never lifted (scene/trial)        lifted, missed")
    for folder, name, n, ok, never, missed in rows:
        fmt = lambda xs: ("-" if not xs else f"{len(xs)}: " + ", ".join(map(str, xs[:8])) + (" ..." if len(xs) > 8 else ""))  # noqa: E731
        print(f"{folder:7s} {name:40s} {ok:>4d}/{n:<4d}  {fmt(never):32s}  {fmt(missed)}")
    tot_never = sum(len(r[4]) for r in rows)
    tot_missed = sum(len(r[5]) for r in rows)
    print(f"\nAll experiments: {tot_never} failures never lifted the cube, {tot_missed} lifted it but missed the target.")


if __name__ == "__main__":
    sys.exit(main())
