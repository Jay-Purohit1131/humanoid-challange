"""Stage 2 of perception: turn one clip's pixel detections into a metric trajectory on the table.

For one clip it:
  1. estimates the table frame from the ArUco marker (one median pose for the clip),
  2. locates the target (ray-plane at z = 0) and checks its size against the real 10 cm square,
  3. locates the cube at the start and end (ray through the top-face centre, plane z = cube height),
  4. computes the pinch point in 3D for every frame (hand-length depth, method A),
  5. corrects the demo's depth scale so the pinch is at the cube's centre height at the grasp,
     and validates that scale at the release,
  6. smooths the trajectory and saves it with the gripper state (closed between grasp and release).

Needs: data/processed/<clip>_2d.npz, <clip>_events.json (detect_events.py), data/calib/camera.npz
Usage: python scripts/build_trajectory.py clean_3 [more clips...]
Output: data/processed/<clip>_traj.npz, outputs/trajectories/<clip>.png (and <clip>_axes.jpg)
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.measurements import CUBE_EDGE_M, HAND_0_9_M, MARKER_LENGTH_M, TARGET_SIDE_M  # noqa: E402
from src.perception import geometry as geo  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
RAW = PROJECT_ROOT / "data" / "raw"
CALIB = PROJECT_ROOT / "data" / "calib" / "camera.npz"
OUT_DIR = PROJECT_ROOT / "outputs" / "trajectories"

ANCHOR_WINDOW_S = 0.3        # median pinch over this window around the grasp / release
GRASP_SEARCH_S = (-0.3, 0.3)   # the hand's lowest point near the detected grasp = the moment of contact
RELEASE_SEARCH_S = (-0.3, 1.0) 
MAX_GAP_S = 0.3              
SMOOTH_S = 0.25              # window for the trajectory
Z_VALID_M = (-0.03, 0.45)    # pinch heights outside this range are impossible -> treated as missing
MAX_JUMP_M = 0.05            # a jump of > 5 cm between frames (3 m/s at 60 fps) is a tracking glitch


def median_px(arr, mask=None):
    a = arr if mask is None else arr[mask]
    a = a.reshape(len(a), -1)
    a = a[~np.isnan(a[:, 0])]
    return np.median(a, axis=0) if len(a) else None


def window(n, centre, half):
    m = np.zeros(n, bool)
    m[max(0, centre - half): min(n, centre + half + 1)] = True
    return m


def fill_and_smooth(x, fps):
    """Interpolate short gaps, then Savitzky-Golay smooth each continuous run (NaN elsewhere)."""

    x = x.copy()
    n = len(x)
    ok = ~np.isnan(x[:, 0])
    idx = np.arange(n)
    max_gap = int(MAX_GAP_S * fps)
    if ok.sum() >= 2:
        filled = np.column_stack([np.interp(idx, idx[ok], x[ok, k]) for k in range(x.shape[1])])
        gap_start = None
        for i, good in enumerate(np.r_[ok, True]):
            if not good and gap_start is None:
                gap_start = i
            elif good and gap_start is not None:
                if 0 < gap_start and i < n and i - gap_start <= max_gap:
                    x[gap_start:i] = filled[gap_start:i]
                gap_start = None
    ok = ~np.isnan(x[:, 0])
    win = max(5, int(SMOOTH_S * fps) | 1)
    start = None
    for i, good in enumerate(np.r_[ok, False]):
        if good and start is None:
            start = i
        elif not good and start is not None:
            if i - start > win:
                x[start:i] = savgol_filter(x[start:i], win, 2, axis=0)
            start = None
    return x


def build(clip):
    d = np.load(PROCESSED / f"{clip}_2d.npz")
    ev = json.load(open(PROCESSED / f"{clip}_events.json"))
    K, dist = geo.load_calibration(CALIB)
    fps, t = float(d["fps"]), d["t"]
    n = len(t)
    if ev["grasp_frame"] is None or ev["release_frame"] is None:
        print(f"{clip}: no grasp/release events, skipping")
        return None
    g, r = ev["grasp_frame"], ev["release_frame"]

    # 1. Table frame
    R, tvec, mk = geo.marker_pose(d["marker_corners"], K, dist, MARKER_LENGTH_M)

    # 2. Target: centre and size from its corners on the table plane (the size is a free accuracy check)
    box_px = np.nanmedian(d["target_box"], axis=0)
    box_tab = geo.ray_plane(box_px, K, dist, R, tvec, 0.0)
    sides = np.linalg.norm(np.roll(box_tab, -1, axis=0) - box_tab, axis=1)
    target = box_tab.mean(axis=0)

    # 3. Cube at the start and end: the top-face centre lies on the plane z = cube height
    first, last = t < t[0] + 1.0, t > t[-1] - 1.0
    top_start = geo.ray_plane(median_px(d["cube_px"], first), K, dist, R, tvec, CUBE_EDGE_M)[0]
    top_end = geo.ray_plane(median_px(d["cube_px"], last), K, dist, R, tvec, CUBE_EDGE_M)[0]
    cube_start = top_start - [0, 0, CUBE_EDGE_M / 2]          # cube centre
    cube_end = top_end - [0, 0, CUBE_EDGE_M / 2]

    # 4. Pinch point in the camera frame (method A, ruler hand length)
    pinch_cam = geo.pinch_cam_hand_length(d["hand_px"], d["pinch_px"], K, dist, HAND_0_9_M)

    # 5. Per-demo scale: scaling the depth by s scales the camera-frame point by s.
    half = int(ANCHOR_WINDOW_S * fps / 2)
    z_anchor = CUBE_EDGE_M / 2
    rt = R.T @ tvec

    def scale_at(frame):
        p = median_px(pinch_cam, window(n, frame, half))
        return None if p is None else (z_anchor + rt[2]) / (R.T @ p)[2]

    def lowest_point(z, frame, search):
        """Frame where the (smoothed) pinch is lowest within a search window around `frame`."""
        lo, hi = max(0, frame + int(search[0] * fps)), min(n, frame + int(search[1] * fps))
        seg = z[lo:hi]
        return None if np.isnan(seg).all() else lo + int(np.nanargmin(seg))

    s_g = scale_at(g)
    if s_g is None:
        print(f"{clip}: no hand tracked at the grasp, skipping")
        return None
    # The cube-based events can be a little off (shadows). Refine them with the hand: when grasping and
    # placing, the pinch reaches its LOWEST point at the moment of contact.
    z1 = fill_and_smooth(geo.to_table(pinch_cam * s_g, R, tvec), fps)[:, 2]
    g = lowest_point(z1, g, GRASP_SEARCH_S) or g
    s_g = scale_at(g) or s_g
    z1 = fill_and_smooth(geo.to_table(pinch_cam * s_g, R, tvec), fps)[:, 2]
    r = lowest_point(z1, r, RELEASE_SEARCH_S) or r

    # Validation with ONE anchor: how high is the pinch at the release if the grasp scale held throughout?
    p_r = median_px(pinch_cam, window(n, r, half))
    one_anchor_release_z = None if p_r is None else geo.to_table(p_r * s_g, R, tvec)[0][2]

    # Two anchors: hand-length depth is pose-sensitive, and the hand's pose when placing differs from when
    # grasping. So the scale is also anchored at the release (pinch at cube-centre height again) and blended
    # linearly in time between the two contacts (grasp scale before, release scale after).
    s_r = scale_at(r) or s_g
    s_t = np.interp(np.arange(n), [g, r], [s_g, s_r])
    pinch_tab = geo.to_table(pinch_cam * s_t[:, None], R, tvec)
    grasp_xyz = geo.to_table(median_px(pinch_cam, window(n, g, half)) * s_g, R, tvec)[0]
    release_xyz = None if p_r is None else geo.to_table(p_r * s_r, R, tvec)[0]
    scale = s_g

    # Reject impossible points, then interpolate short gaps and smooth. Gripper state, save.
    bad = (pinch_tab[:, 2] < Z_VALID_M[0]) | (pinch_tab[:, 2] > Z_VALID_M[1])
    jump = np.r_[False, np.linalg.norm(np.diff(pinch_tab, axis=0), axis=1) > MAX_JUMP_M]
    bad |= jump | np.r_[jump[1:], False]
    n_rejected = int((bad & ~np.isnan(pinch_tab[:, 0])).sum())
    pinch_tab[bad] = np.nan
    pinch_tab = fill_and_smooth(pinch_tab, fps)
    gripper = np.zeros(n)
    gripper[g:r] = 1.0                                        # 1 = closed

    checks = {
        "grasp_t_cube": ev["grasp_t"], "grasp_t_hand": round(float(t[g]), 2),
        "release_t_cube": ev["release_t"], "release_t_hand": round(float(t[r]), 2),
        "marker_frames": mk["frames"], "marker_reproj_px": round(mk["reproj_px"], 3),
        "marker_alt_reproj_px": None if mk["alt_reproj_px"] is None else round(mk["alt_reproj_px"], 2),
        "target_sides_cm": np.round(sides * 100, 2).tolist(),
        "target_side_error_cm": round(float(np.mean(sides) - TARGET_SIDE_M) * 100, 2),
        "depth_scale": round(float(scale), 3), "depth_scale_release": round(float(s_r), 3),
        "one_anchor_release_z_cm": None if one_anchor_release_z is None else round(float(one_anchor_release_z) * 100, 2),
        "grasp_xy_offset_cm": round(float(np.linalg.norm(grasp_xyz[:2] - cube_start[:2])) * 100, 2),
        "release_z_error_one_anchor_cm": None if one_anchor_release_z is None else
            round(float(one_anchor_release_z - z_anchor) * 100, 2),
        "release_xy_offset_cm": None if release_xyz is None else
            round(float(np.linalg.norm(release_xyz[:2] - cube_end[:2])) * 100, 2),
        "carry_cm": round(float(np.linalg.norm(cube_end[:2] - cube_start[:2])) * 100, 1),
        "points_rejected": n_rejected,
    }
    np.savez(PROCESSED / f"{clip}_traj.npz", t=t, fps=fps, pinch=pinch_tab, gripper=gripper,
             grasp_frame=g, release_frame=r, cube_start=cube_start, cube_end=cube_end,
             target=target, target_corners=box_tab, scale=s_t, checks=json.dumps(checks))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(clip, t, pinch_tab, g, r, cube_start, cube_end, box_tab)
    save_axes_image(clip, K, dist, R, tvec, n)
    return checks


def plot(clip, t, p, g, r, cube_start, cube_end, box):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for k, name in enumerate("xyz"):
        ax1.plot(t, p[:, k] * 100, label=name)
    ax1.axvline(t[g], color="green", lw=2, label="grasp")
    ax1.axvline(t[r], color="red", lw=2, label="release")
    ax1.axhline(CUBE_EDGE_M / 2 * 100, color="grey", ls=":", label="cube centre height")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("table frame (cm)")
    ax1.set_title(f"{clip}: pinch trajectory")
    ax1.legend(fontsize=8)
    sq = np.vstack([box, box[:1]])
    ax2.plot(sq[:, 0] * 100, sq[:, 1] * 100, color="green", label="target")
    ax2.plot(p[:, 0] * 100, p[:, 1] * 100, color="black", lw=1, label="pinch path")
    ax2.plot(*(p[g:r, :2] * 100).T, color="orange", lw=2, label="holding the cube")
    ax2.scatter(*cube_start[:2] * 100, s=120, marker="s", color="royalblue", label="cube start")
    ax2.scatter(*cube_end[:2] * 100, s=120, marker="s", facecolor="none", edgecolor="royalblue", label="cube end")
    ax2.scatter(0, 0, marker="x", color="red", s=80, label="marker (origin)")
    ax2.set_aspect("equal")
    ax2.set_xlabel("x (cm)")
    ax2.set_ylabel("y (cm)")
    ax2.set_title("top-down view, table frame")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{clip}.png", dpi=110)
    plt.close(fig)


def save_axes_image(clip, K, dist, R, tvec, n):
    video = next((p for p in [RAW / f"{clip}.mp4", RAW / f"{clip}.mov", RAW / f"{clip}.MOV"] if p.exists()), None)
    if video is None:
        return
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(OUT_DIR / f"{clip}_axes.jpg"), geo.draw_table_axes(frame, K, dist, R, tvec, 0.08))


if __name__ == "__main__":
    clips = sys.argv[1:] or ["clean_3"]
    for c in clips:
        ch = build(c)
        if ch is None:
            continue
        print(f"\n{c}")
        print(f"  marker: {ch['marker_frames']} frames, reprojection {ch['marker_reproj_px']} px "
              f"(alternative pose {ch['marker_alt_reproj_px']} px)")
        print(f"  target sides: {ch['target_sides_cm']} cm  -> mean error {ch['target_side_error_cm']:+.2f} cm vs "
              f"{TARGET_SIDE_M * 100:.0f} cm   [table-frame accuracy check]")
        print(f"  carry: {ch['carry_cm']} cm   depth scale: grasp {ch['depth_scale']}, release "
              f"{ch['depth_scale_release']}  (1.0 = ruler hand length was right)")
        print(f"  rejected {ch['points_rejected']} impossible points (depth blow-ups / glitches)")
        print(f"  events:  grasp {ch['grasp_t_cube']} s (cube) -> {ch['grasp_t_hand']} s (hand lowest point);  "
              f"release {ch['release_t_cube']} s -> {ch['release_t_hand']} s")
        print(f"  grasp:   pinch {ch['grasp_xy_offset_cm']} cm from the cube centre (horizontal)")
        print(f"  release: with the grasp scale alone the pinch would be at {ch['one_anchor_release_z_cm']} cm "
              f"(should be {CUBE_EDGE_M / 2 * 100:.2f}): error {ch['release_z_error_one_anchor_cm']:+.2f} cm "
              f"-> corrected by the release anchor")
        print(f"           horizontal: pinch {ch['release_xy_offset_cm']} cm from the cube centre at the release")
    print(f"\nTrajectories: {PROCESSED}/<clip>_traj.npz   Plots: {OUT_DIR}/")
