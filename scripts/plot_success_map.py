"""Success map: where in the workspace does one demo still work?

Reads the per-scene results of evaluate_generalisation.py and plots, for each scene, how far the cube
and the target were moved from their recorded positions, coloured by success. Also prints the success
rate by how far the objects moved.

Usage: python scripts/plot_success_map.py outputs/eval/clean_12_object_relative.json [more .json files]
Output: <same name>_map.png next to each .json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BINS_CM = [0, 4, 8, 12, 16, 25, 40]


def plot(path):
    with open(path) as f:
        res = json.load(f)
    ok = np.array([r["success"] for r in res])
    cube = np.array([r["cube_offset"] for r in res]) * 100
    target = np.array([r["target_offset"] for r in res]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, xy, name in [(axes[0], cube, "cube"), (axes[1], target, "target")]:
        ax.scatter(*xy[ok].T, c="tab:green", s=50, label=f"success ({ok.sum()})")
        ax.scatter(*xy[~ok].T, c="tab:red", marker="x", s=60, label=f"failure ({(~ok).sum()})")
        ax.scatter(0, 0, marker="*", s=250, c="black", label="recorded position")
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{name} moved in x (cm)")
        ax.set_ylabel(f"{name} moved in y (cm)")
        ax.set_title(f"{name} position vs outcome")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"{Path(path).stem}: {ok.sum()}/{len(ok)} successful "
                 f"(sim robot base is at x = -0.56 m; +x is away from the robot)")
    plt.tight_layout()
    out = Path(path).with_name(Path(path).stem + "_map.png")
    plt.savefig(out, dpi=120)
    plt.close(fig)

    print(f"\n{Path(path).stem}: {ok.sum()}/{len(ok)} = {100 * ok.mean():.0f}%   map -> {out}")
    print("  success rate by how far the CUBE moved, and by how far the TARGET moved:")
    shift = {"cube": np.linalg.norm(cube, axis=1), "target": np.linalg.norm(target, axis=1)}
    for lo, hi in zip(BINS_CM[:-1], BINS_CM[1:]):
        row = f"    {lo:2d}-{hi:2d} cm:"
        for k in ("cube", "target"):
            m = (shift[k] >= lo) & (shift[k] < hi)
            row += f"   {k} {ok[m].sum():3d}/{m.sum():<3d}" if m.any() else f"   {k}   -    "
        print(row)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        plot(p)
