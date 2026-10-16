"""My phone video next to the simulated Panda replaying it: the figure that shows my data driving the robot.

Left: the recorded clip with the tracked hand, the pinch point (green = open, red = holding) and the table
frame from the ArUco marker. Right: the Panda following the trajectory extracted from that same clip, with
the cube and target at their recorded positions. Both run on the same clock (sim step k <-> video time
t_start + k / 20 Hz).

Usage:  python scripts/make_side_by_side.py clean_12 [--camera birdview|frontview] [--height 270]
Output: docs/media/human_vs_robot.gif
"""
import argparse
import sys
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.measurements import MARKER_LENGTH_M  # noqa: E402
from src.perception import geometry as geo  # noqa: E402
from src.sim.cube_env import CubeToTarget  # noqa: E402
from src.sim.replay import CONTROL_HZ, PREROLL_TOL, SETTLE_STEPS, contact_fixes, load_demo, move_action  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
CALIB = PROJECT_ROOT / "data" / "calib" / "camera.npz"
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def sim_frames(clip, camera, size):
    """Replay the clip's trajectory at its recorded positions; one frame per control step from the path start."""
    demo = load_demo(clip)
    path, _ = contact_fixes(demo["path"], demo["grip"], demo["cube"][:2], demo["target"][:2])
    cube, target = demo["cube"], demo["target"]
    env = CubeToTarget(robots="Panda", has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
                       camera_names=camera, camera_heights=size, camera_widths=size, control_freq=CONTROL_HZ,
                       cube_x_range=(cube[0], cube[0]), cube_y_range=(cube[1], cube[1]),
                       target_xy=(target[0], target[1]), seed=0)
    obs = env.reset()
    for _ in range(10):                                         # cube settles
        obs, _, _, _ = env.step(np.r_[np.zeros(6), -1.0])
    for _ in range(200):                                        # move to the start of the path (not shown)
        if np.linalg.norm(path[0] - obs["robot0_eef_pos"]) < PREROLL_TOL:
            break
        obs, _, _, _ = env.step(move_action(obs["robot0_eef_pos"], path[0], False))
    frames = []
    for k in range(len(path)):
        obs, _, _, _ = env.step(move_action(obs["robot0_eef_pos"], path[k], demo["grip"][k]))
        frames.append(obs[f"{camera}_image"].copy())
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(move_action(obs["robot0_eef_pos"], obs["robot0_eef_pos"], False))
        frames.append(obs[f"{camera}_image"].copy())
    success = bool(env._check_success())
    env.close()
    return frames, demo["t"][0], success


def video_frames(clip, times, height):
    """The clip's frames nearest to `times`, with hand, pinch and table axes drawn (RGB)."""
    video = next((p for p in [RAW / f"{clip}.mp4", RAW / f"{clip}.mov", RAW / f"{clip}.MOV"] if p.exists()), None)
    if video is None:
        sys.exit(f"No video for {clip} in {RAW}")
    d = np.load(PROCESSED / f"{clip}_2d.npz")
    traj = np.load(PROCESSED / f"{clip}_traj.npz")
    K, dist = geo.load_calibration(CALIB)
    R, tvec, _ = geo.marker_pose(d["marker_corners"], K, dist, MARKER_LENGTH_M)
    idx = np.clip(np.searchsorted(d["t"], times), 0, len(d["t"]) - 1)
    wanted = set(idx.tolist())
    cap, grabbed, i = cv2.VideoCapture(str(video)), {}, 0
    while len(grabbed) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            geo.draw_table_axes(frame, K, dist, R, tvec, 0.08)
            hand = d["hand_px"][i]
            if not np.isnan(hand[0, 0]):
                for a, b in HAND_LINKS:
                    cv2.line(frame, tuple(hand[a].astype(int)), tuple(hand[b].astype(int)), (255, 255, 255), 3)
                holding = traj["gripper"][i] > 0.5
                cv2.circle(frame, tuple(d["pinch_px"][i].astype(int)), 22, (0, 0, 255) if holding else (0, 200, 0), -1)
            h, w = frame.shape[:2]
            grabbed[i] = cv2.cvtColor(cv2.resize(frame, (int(w * height / h), height)), cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    last = grabbed[max(grabbed)]
    return [grabbed.get(j, last) for j in idx]


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default="clean_12")
    ap.add_argument("--camera", default="birdview", help="robosuite camera: birdview (from above) or frontview")
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--every", type=int, default=2, help="keep every Nth control step (2 -> 10 fps)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "docs" / "media" / "human_vs_robot.gif"))
    args = ap.parse_args()

    sim, t_start, success = sim_frames(args.clip, args.camera, args.height)
    sim = sim[::args.every]
    times = t_start + np.arange(len(sim)) * args.every / CONTROL_HZ          # same clock as the replay
    vid = video_frames(args.clip, times, args.height)
    frames = [np.hstack([label(v, f"my phone video ({args.clip})"), label(s, "Panda driven by that video")])
              for v, s in zip(vid, sim)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, frames, duration=1000 * args.every / CONTROL_HZ, loop=0)
    print(f"{len(frames)} frames, replay success {success} -> {args.out}")
