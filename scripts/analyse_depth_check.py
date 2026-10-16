"""Depth check: how accurately can one camera recover the height of the pinch point?

Compares two ways of getting metric depth from a monocular video:
  Method A, hand length:  Z = f x (real wrist->middle-finger-base length) / (its length in pixels)
  Method B, solvePnP:     fit MediaPipe's 3D hand shape (world landmarks, rescaled to my hand)
                          to the 2D landmarks, giving the hand's full 3D pose.
Each pinch position is converted to the table frame defined by the ArUco marker, and its height
is compared with the known height of the surface the fingertips were resting on.

Needs:  data/processed/depth_check_2d.npz   (run scripts/extract_2d.py on the depth-check clip first)
        data/calib/camera.npz                (run scripts/calibrate_camera.py first)
Usage:  python scripts/analyse_depth_check.py [path/to/depth_check_2d.npz]
"""
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIB_PATH = PROJECT_ROOT / "data" / "calib" / "camera.npz"
OUT_DIR = PROJECT_ROOT / "outputs" / "depth_check"

# ---- MEASURED VALUES: replace with your ruler measurements (metres)
MARKER_LENGTH_M = 0.100        # ArUco marker, outer edge of the black square
HAND_0_9_M = 0.110             # wrist crease -> base of the middle finger (MediaPipe 0 -> 9)
CUBE_EDGE_M = 0.05
BOOK_HEIGHT_M = 0.100
FINGERTIP_OFFSET_M = 0.015   # pinch midpoint sits ~half a fingertip above the surface it rests on

# ---- The holds, IN THE ORDER YOU RECORDED THEM: (label, surface height, tilted hand?)
'''HOLDS = [
    ("table near",        0.0,           False),
    ("cube top",          CUBE_EDGE_M,   False),
    ("book stack",        BOOK_HEIGHT_M, False),
    ("table middle",      0.300,           False),
    ("table far",         0.600,           False),
    
]
'''
HOLDS = [
    ("table (on target)",        0.0,           False),
    ("book stack",               BOOK_HEIGHT_M, False),
    ("book stack",               BOOK_HEIGHT_M, False),
    ("book stack, hand upright", BOOK_HEIGHT_M, True),
]
# ---- Hold detection. A frame counts as "still" if, over a short window around it, the pinch point
# stays within STILL_RADIUS_PX of the window's median. 
WINDOW_S = 0.5                 # window length for the stillness test
STILL_RADIUS_PX = 20.0         # how far the pinch may wander within the window (pixels)
MIN_PRESENT = 0.8              # the hand must be detected in at least 80% of the window
MIN_HOLD_S = 1.0               # a hold must last at least this long
HOLD_CORE = 0.6                # use the middle 60% of each hold
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "depth_check.mp4"   

WRIST, MIDDLE_BASE, THUMB_TIP, INDEX_TIP = 0, 9, 4, 8


def marker_pose(corners_all, K, dist):
    """One marker pose for the clip, from the median of the detected corner positions."""

    detected = ~np.isnan(corners_all[:, 0, 0])
    if detected.sum() == 0:
        sys.exit("The marker was never detected in this clip.")
    corners = np.median(corners_all[detected], axis=0).astype(np.float64)
    h = MARKER_LENGTH_M / 2
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left; marker z points up out of the table
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        sys.exit("solvePnP failed for the marker.")
    R, _ = cv2.Rodrigues(rvec)
    print(f"Marker: {detected.sum()} frames detected; camera {np.linalg.norm(tvec) * 100:.1f} cm from marker")
    return R, tvec.ravel()


def to_table(p_cam, R_m, t_m):
    """Camera-frame point(s) -> table (marker) frame. z = height above the table."""

    return (np.atleast_2d(p_cam) - t_m) @ R_m      # equals R_m.T @ (p - t) for each row


def undistort(px, K, dist):
    """Distorted pixel(s) -> ideal pinhole pixel(s)."""

    return cv2.undistortPoints(np.asarray(px, np.float64).reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)


def method_a(hand_px, pinch_px, K, dist):
    """Hand-length depth: Z = f * real length / pixel length, then back-project the pinch pixel."""

    p = undistort(np.vstack([hand_px[WRIST], hand_px[MIDDLE_BASE], pinch_px]), K, dist)
    length_px = np.linalg.norm(p[0] - p[1])
    f = (K[0, 0] + K[1, 1]) / 2
    Z = f * HAND_0_9_M / length_px
    x = (p[2, 0] - K[0, 2]) / K[0, 0]
    y = (p[2, 1] - K[1, 2]) / K[1, 1]
    return Z * np.array([x, y, 1.0])


def method_b(hand_px, hand_world, K, dist):
    """solvePnP: fit the (rescaled) 3D hand shape to the 2D landmarks; return the 3D pinch point."""

    scale = HAND_0_9_M / np.linalg.norm(hand_world[WRIST] - hand_world[MIDDLE_BASE])
    obj = (hand_world * scale).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, hand_px.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return np.full(3, np.nan)
    R, _ = cv2.Rodrigues(rvec)
    pinch_hand = (obj[THUMB_TIP] + obj[INDEX_TIP]) / 2
    return R @ pinch_hand + tvec.ravel()


def find_holds(pinch_px, fps):
    """Segments where the pinch point stays within STILL_RADIUS_PX for at least MIN_HOLD_S."""

    n, half = len(pinch_px), max(1, int(WINDOW_S * fps / 2))
    present = ~np.isnan(pinch_px[:, 0])
    still = np.zeros(n, bool)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = pinch_px[lo:hi][present[lo:hi]]
        if present[i] and len(window) >= MIN_PRESENT * (hi - lo):
            centre = np.median(window, axis=0)
            still[i] = np.max(np.linalg.norm(window - centre, axis=1)) < STILL_RADIUS_PX
    holds, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if (i - start) / fps >= MIN_HOLD_S:
                holds.append((start, i))
            start = None
    return holds


def save_hold_snapshots(holds, pinch_px, fps):
    """One image per detected hold (middle frame, pinch circled), so each hold can be labelled."""
    
    if not VIDEO_PATH.exists():
        return
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    for k, (s, e) in enumerate(holds):
        mid = (s + e) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.circle(frame, tuple(int(v) for v in pinch_px[mid]), 25, (0, 0, 255), 4)
        cv2.putText(frame, f"hold {k + 1}: {s / fps:.1f}-{e / fps:.1f} s", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
        cv2.imwrite(str(OUT_DIR / f"hold_{k + 1}.jpg"), cv2.resize(frame, (960, 540)))
    cap.release()


def main(npz_path):
    calib = np.load(CALIB_PATH)
    K, dist = calib["K"], calib["dist"]
    d = np.load(npz_path)
    fps, t = float(d["fps"]), d["t"]
    hand_px, hand_world, pinch_px = d["hand_px"], d["hand_world"], d["pinch_px"]
    print(f"{npz_path.name}: {len(t)} frames, {fps:.1f} fps; calibration RMS {float(calib['rms']):.2f} px")

    R_m, t_m = marker_pose(d["marker_corners"], K, dist)

    # Height of the pinch point in every frame, by both methods
    height_a = np.full(len(t), np.nan)
    height_b = np.full(len(t), np.nan)
    for i in range(len(t)):
        if np.isnan(pinch_px[i, 0]):
            continue
        height_a[i] = to_table(method_a(hand_px[i], pinch_px[i], K, dist), R_m, t_m)[0, 2]
        height_b[i] = to_table(method_b(hand_px[i], hand_world[i], K, dist), R_m, t_m)[0, 2]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holds = find_holds(pinch_px, fps)
    save_hold_snapshots(holds, pinch_px, fps)
    print(f"\nFound {len(holds)} holds (expected {len(HOLDS)}); snapshots in {OUT_DIR}/hold_N.jpg:")
    for k, (s, e) in enumerate(holds):
        print(f"  hold {k + 1}: {t[s]:5.1f}-{t[e - 1]:5.1f} s")
    if len(holds) != len(HOLDS):
        print("\n  -> The number of holds doesn't match HOLDS. Check the times against your recording;")
        print("     adjust STILL_RADIUS_PX / MIN_HOLD_S, or edit HOLDS to match what you actually did.")
    n = min(len(holds), len(HOLDS))

    rows = []
    for (s, e), (label, surface, tilted) in zip(holds[:n], HOLDS[:n]):
        pad = int((e - s) * (1 - HOLD_CORE) / 2)
        core = slice(s + pad, e - pad)
        truth = surface + FINGERTIP_OFFSET_M
        est_a, est_b = np.nanmedian(height_a[core]), np.nanmedian(height_b[core])
        rows.append({"hold": label, "tilted": tilted, "true_cm": truth * 100,
                     "A_cm": est_a * 100, "B_cm": est_b * 100,
                     "A_err_cm": (est_a - truth) * 100, "B_err_cm": (est_b - truth) * 100,
                     "start_s": t[s], "end_s": t[e - 1]})

    print(f"\n{'hold':20s} {'true':>6s} {'A (len)':>8s} {'B (PnP)':>8s} {'A err':>7s} {'B err':>7s}   (cm)")
    for r in rows:
        print(f"{r['hold']:20s} {r['true_cm']:6.1f} {r['A_cm']:8.1f} {r['B_cm']:8.1f} "
              f"{r['A_err_cm']:+7.1f} {r['B_err_cm']:+7.1f}")

    def summary(method, tilted):
        errs = np.array([r[f"{method}_err_cm"] for r in rows if r["tilted"] == tilted])
        return errs
    print("\nSummary (cm)                   bias    std    |err| tilted")
    for m, name in [("A", "Hand length 0->9"), ("B", "solvePnP + world landmarks")]:
        flat, tilt = summary(m, False), summary(m, True)
        bias = np.mean(flat) if len(flat) else np.nan
        std = np.std(flat) if len(flat) > 1 else np.nan
        tilt_err = np.mean(np.abs(tilt)) if len(tilt) else np.nan
        print(f"  {name:28s} {bias:+6.2f} {std:6.2f} {tilt_err:8.2f}")

    with open(OUT_DIR / "depth_check_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["hold"])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, height_a * 100, lw=1, label="A: hand length 0->9")
    ax.plot(t, height_b * 100, lw=1, label="B: solvePnP + world landmarks")
    for r in rows:
        ax.axvspan(r["start_s"], r["end_s"], color="grey", alpha=0.15)
        ax.hlines(r["true_cm"], r["start_s"], r["end_s"], colors="black", lw=2)
        ax.text(r["start_s"], r["true_cm"] + 1, r["hold"], fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pinch height above table (cm)")
    ax.set_title("Depth check: estimated vs true pinch height (black bars = truth, grey = holds)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "depth_check.png", dpi=130)
    print(f"\nSaved {OUT_DIR / 'depth_check_results.csv'} and depth_check.png")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "processed" / "depth_check_2d.npz"
    main(path)