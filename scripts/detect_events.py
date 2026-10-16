"""Grasp and release detection from the cube's motion.

The fingers aren't used: on a 5.7 cm cube a pinch never "closes", and fingertips are often hidden.
Instead the cube itself says when it was picked up and put down:
  - its centre moves in the image when it's carried sideways, and
  - its top face looks BIGGER when it's lifted as it gets closer to the camera, which catches
    the vertical lift that barely moves the centre.

Release  = the start of the cube's final rest on the table at its end position.
Grasp    = the end of the cube's LAST rest on the table before that, away from the end position.
           (Taking the last rest, not the first movement, handles re-grasps: putting the cube
           down and picking it up again. The naive "first movement" is also reported.)

Usage:  python scripts/detect_events.py <clip>
        python scripts/detect_events.py data/processed/clean_3_2d.npz
Output: data/processed/<clip>_events.json, outputs/segmentation/<clip>.png, a summary table.
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
PLOT_DIR = PROJECT_ROOT / "outputs" / "segmentation"

TARGET_SIDE_CM = 10.0       
MAX_GAP_S = 0.25             
SMOOTH_FRAMES = 5            
REST_WINDOW_S = 0.3          
REST_RADIUS_CM = 1.0         
LIFT_TOL = 0.04             
END_RADIUS_CM = 2.5          
NAIVE_MOVE_CM = 3.0          
NAIVE_LIFT_RATIO = 1.08      
NAIVE_HOLD_S = 0.2           
MIN_CARRY_CM = 10.0          
MAX_PINCH_TO_CUBE_CM = 5.0   


def fill_short_gaps(x, max_gap):
    """Linearly interpolate NaN runs of length less than or equal to max_gap. Longer gaps stay NaN."""

    x = x.copy()
    x2 = x.reshape(len(x), -1)
    idx = np.arange(len(x2))
    for c in range(x2.shape[1]):
        col = x2[:, c]
        nan = np.isnan(col)
        if nan.all() or not nan.any():
            continue
        filled = np.interp(idx, idx[~nan], col[~nan])
        run_start = None
        for i, is_nan in enumerate(np.r_[nan, False]):
            if is_nan and run_start is None:
                run_start = i
            elif not is_nan and run_start is not None:
                if i - run_start <= max_gap and run_start > 0 and i < len(col):
                    col[run_start:i] = filled[run_start:i]      # only gaps with data on both sides
                run_start = None
        x2[:, c] = col
    return x2.reshape(x.shape)


def smooth(x, size):
    """Median filter that leaves NaN where the input was NaN."""

    nan = np.isnan(x.reshape(len(x), -1)[:, 0])
    if nan.all():
        return x
    filled = fill_short_gaps(x, len(x))                       # temporary: fill everything
    filled = np.where(np.isnan(filled), np.nanmedian(x, axis=0), filled)
    out = median_filter(filled, size=(size,) + (1,) * (x.ndim - 1), mode="nearest")
    out[nan] = np.nan
    return out


def pixels_per_cm(target_box):
    """Scale for this clip, from the target square's apparent area (it's 10 x 10 cm)."""

    ok = ~np.isnan(target_box[:, 0, 0])
    areas = [cv2.contourArea(b.astype(np.float32)) for b in target_box[ok]]
    return np.sqrt(np.median(areas)) / TARGET_SIDE_CM


def runs(mask):
    """ndex pairs of consecutive True runs."""

    out, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def detect(npz_path):
    d = np.load(npz_path)
    clip = npz_path.name.replace("_2d.npz", "")
    fps, t = float(d["fps"]), d["t"]
    n = len(t)
    max_gap = int(MAX_GAP_S * fps)

    cube = smooth(fill_short_gaps(d["cube_px"], max_gap), SMOOTH_FRAMES)
    area = smooth(fill_short_gaps(np.where(d["cube_area"] > 0, d["cube_area"], np.nan), max_gap), SMOOTH_FRAMES)
    pinch = d["pinch_px"]
    ppc = pixels_per_cm(d["target_box"])

    first_s, last_s = t < t[0] + 1.0, t > t[-1] - 1.0
    start, end = np.nanmedian(cube[first_s], axis=0), np.nanmedian(cube[last_s], axis=0)
    area_ratio = area / np.nanmedian(area[first_s])
    disp = np.linalg.norm(cube - start, axis=1) / ppc            
    to_end = np.linalg.norm(cube - end, axis=1) / ppc            

    # Lifted or not, relative to the cube's resting size at each end of the carry
    end_ratio = area / np.nanmedian(area[last_s])
    half = max(1, int(REST_WINDOW_S * fps / 2))
    still = np.zeros(n, bool)
    for i in range(n):
        if np.isnan(cube[i, 0]):
            continue
        w = cube[max(0, i - half): i + half + 1]
        w = w[~np.isnan(w[:, 0])]
        if len(w) >= half:
            still[i] = np.max(np.linalg.norm(w - np.median(w, axis=0), axis=1)) / ppc < REST_RADIUS_CM
    not_lifted_start = area_ratio < 1 + LIFT_TOL
    not_lifted_end = end_ratio < 1 + LIFT_TOL
    rest = still & np.where(to_end < END_RADIUS_CM, not_lifted_end, not_lifted_start)   

    warnings = []
    # Release (touch-down): from this frame on, every detected frame has the cube at its end position
    # and not lifted. Frames where the cube isn't detected don't count against it.
    detected = ~np.isnan(cube[:, 0]) & ~np.isnan(end_ratio)
    violates = detected & ((to_end >= END_RADIUS_CM) | ~not_lifted_end)
    if not detected[-int(0.5 * fps):].any():
        warnings.append("cube not visible at the end of the clip")
    last_bad = np.flatnonzero(violates)
    release = int(last_bad[-1] + 1) if len(last_bad) else None
    if release is not None and release >= n - int(0.3 * fps):
        warnings.append("cube never settled at the end position")
        release = None

    # Grasp: the last frame, before the release, where the cube was still, on the table (not lifted)
    # and away from the end position, i.e. the moment it left its (final) pick-up spot.
    grasp = None
    if release is not None:
        pickup = still & not_lifted_start & (to_end > MIN_CARRY_CM / 2)
        idx = np.flatnonzero(pickup[:release])
        if len(idx):
            grasp = int(idx[-1] + 1)
            gap = np.flatnonzero(detected[grasp:release])
            if len(gap) and gap[0] / fps > 0.5:
                warnings.append(f"cube not visible for {gap[0] / fps:.1f} s after the grasp (timing uncertain)")
        else:
            warnings.append("no on-table pick-up position found before the release")

    # Naive grasp, for comparison: first sustained departure from the start position
    left = (disp > NAIVE_MOVE_CM) | (area_ratio > NAIVE_LIFT_RATIO)
    hold = int(NAIVE_HOLD_S * fps)
    naive = next((r[0] for r in runs(left) if r[1] - r[0] >= hold), None)
    if naive is not None:   # walk back to where the cube was still essentially at rest
        while naive > 0 and (disp[naive - 1] > REST_RADIUS_CM or area_ratio[naive - 1] > 1 + LIFT_TOL):
            naive -= 1

    carry = np.linalg.norm(end - start) / ppc
    if carry < MIN_CARRY_CM:
        warnings.append(f"cube only travelled {carry:.1f} cm")
    if grasp is not None:
        w = pinch[grasp: grasp + int(0.3 * fps)]             
        w = w[~np.isnan(w[:, 0])]
        if len(w) == 0:
            warnings.append("no hand detected at the grasp")
        else:
            gap = np.linalg.norm(np.median(w, axis=0) - cube[grasp]) / ppc
            if gap > MAX_PINCH_TO_CUBE_CM:
                warnings.append(f"pinch {gap:.1f} cm from the cube at the grasp")
    box = np.nanmedian(d["target_box"], axis=0).astype(np.float32)
    if cv2.pointPolygonTest(box, (float(end[0]), float(end[1])), False) < 0:
        warnings.append("end position is outside the target square")
    if grasp is not None and release is not None and not (0.5 <= t[release] - t[grasp] <= 15):
        warnings.append(f"implausible carry time {t[release] - t[grasp]:.1f} s")
    regrasp = grasp is not None and naive is not None and t[grasp] - t[naive] > 0.5

    events = {
        "clip": clip, "fps": fps, "px_per_cm": round(float(ppc), 2),
        "grasp_frame": grasp, "release_frame": release,
        "grasp_t": None if grasp is None else round(float(t[grasp]), 3),
        "release_t": None if release is None else round(float(t[release]), 3),
        "naive_grasp_t": None if naive is None else round(float(t[naive]), 3),
        "regrasp_detected": bool(regrasp),
        "cube_start_px": np.round(start, 1).tolist(), "cube_end_px": np.round(end, 1).tolist(),
        "carry_cm": round(float(carry), 1), "warnings": warnings,
    }
    with open(PROCESSED / f"{clip}_events.json", "w") as f:
        json.dump(events, f, indent=2)

    # Plot: distance from start / to end, apparent size, with the detected events
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(t, disp, label="distance from start (cm)")
    ax1.plot(t, to_end, label="distance to end (cm)")
    ax1.fill_between(t, 0, np.nanmax(disp) if np.isfinite(np.nanmax(disp)) else 1, where=rest,
                     color="grey", alpha=0.15, label="on the table, at rest")
    ax2.plot(t, area_ratio, color="purple", label="apparent size vs start (lifted = bigger)")
    ax2.axhline(1, color="grey", lw=0.8)
    for ax in (ax1, ax2):
        if grasp is not None:
            ax.axvline(t[grasp], color="green", lw=2, label="grasp" if ax is ax1 else None)
        if release is not None:
            ax.axvline(t[release], color="red", lw=2, label="release" if ax is ax1 else None)
        if regrasp:
            ax.axvline(t[naive], color="green", ls="--", lw=1, label="naive grasp (first lift)" if ax is ax1 else None)
    ax1.set_ylabel("cm")
    ax2.set_ylabel("size ratio")
    ax2.set_xlabel("time (s)")
    ax1.set_title(f"{clip}: grasp/release from cube motion" + ("  [RE-GRASP]" if regrasp else ""))
    ax1.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / f"{clip}.png", dpi=110)
    plt.close(fig)
    return events


def main(paths):
    print(f"{'clip':12s} {'grasp s':>8s} {'release s':>9s} {'carry cm':>9s} {'naive s':>8s}  notes")
    for p in paths:
        e = detect(Path(p))
        def fmt(v):
            return "   -" if v is None else f"{v:8.2f}"
        notes = "; ".join((["RE-GRASP"] if e["regrasp_detected"] else []) + e["warnings"])
        print(f"{e['clip']:12s} {fmt(e['grasp_t'])} {fmt(e['release_t']):>9s} {e['carry_cm']:9.1f} "
              f"{fmt(e['naive_grasp_t'])}  {notes or 'ok'}")
    print(f"\nEvents: {PROCESSED}/<clip>_events.json   Plots: {PLOT_DIR}/<clip>.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(f for pattern in ["clean_*_2d.npz", "messy_*_2d.npz", "rotation_*_2d.npz"]
                       for f in glob.glob(str(PROCESSED / pattern)))
    if not files:
        sys.exit(f"No *_2d.npz files found in {PROCESSED}. Run extract_2d.py first.")
    main(files)