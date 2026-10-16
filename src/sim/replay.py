"""Shared replay machinery: load a recorded demo into sim coordinates, and run one episode.

Used by scripts/replay_in_sim.py for replay at the recorded positions and by
scripts/evaluate_generalisation.py for raw vs object-relative replay on random scenes.
"""
from pathlib import Path

import numpy as np
import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"        # images are right way up
from src.sim.cube_env import CubeToTarget  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"

SIM_TABLE_Z = 0.8                 # robosuite table top height
WORKSPACE_CENTRE = np.array([0.0, 0.0])   # where the middle of the recorded carry is placed (sim x, y)
CONTROL_HZ = 20
MAX_STEP = 0.05                   # metres per action unit (OSC_POSE)
GAIN = 1.0                        # proportional gain while tracking the trajectory
PREROLL_TOL = 0.01                # reach the first waypoint within 1 cm before replaying
LIFT_THRESHOLD = 0.05
SETTLE_STEPS = 20                 # after the trajectory ends, let the cube settle before scoring
CAMERA = "frontview"

""" ---- Contact fixes: The hand's estimated height is least reliable close to
 contact, and the robot's fingers are wider than a recorded pinch, so near the cube and target the replay:"""
CUBE_HALF = 0.0285
MIN_Z = CUBE_HALF - 0.005          # never aim below the cube's mid-height minus 5 mm
CLEAR_RADIUS = 0.07                
CLEARANCE = 0.03                   
VERTICAL_STEPS = 12                # straight-down approach (0.6 s) into the grasp and the release


# -----Mapping demo coordinates to sim
def make_mapping(cube_start, target):
    """Table frame -> sim frame: rotate about z so the carry (cube to target) points along +y,
    and shift the middle of the carry to WORKSPACE_CENTRE."""
    carry = target[:2] - cube_start[:2]
    angle = np.pi / 2 - np.arctan2(carry[1], carry[0])        # rotate the carry direction onto +y
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s], [s, c]])
    mid = (cube_start[:2] + target[:2]) / 2

    def to_sim(p):
        p = np.atleast_2d(p)
        xy = (p[:, :2] - mid) @ Rz.T + WORKSPACE_CENTRE
        return np.column_stack([xy, p[:, 2] + SIM_TABLE_Z])
    return to_sim, {"rotation_deg": float(np.degrees(angle)), "centre_table_xy": mid.tolist()}


def resample(t, pinch, gripper, hz):
    """Pinch positions and gripper state on a regular 1/hz grid (interpolating over tracking gaps)."""
    ok = ~np.isnan(pinch[:, 0])
    t_new = np.arange(t[ok][0], t[ok][-1], 1.0 / hz)
    p = np.column_stack([np.interp(t_new, t[ok], pinch[ok, k]) for k in range(3)])
    g = np.interp(t_new, t, gripper) > 0.5
    return t_new, p, g


def load_demo(clip):
    """A recorded demo in sim coordinates at 20 Hz: path, gripper, and the cube/target it was recorded with."""
    d = np.load(PROCESSED / f"{clip}_traj.npz")
    to_sim, mapping = make_mapping(d["cube_start"], d["target"])
    t, path, grip = resample(d["t"], d["pinch"], d["gripper"], CONTROL_HZ)
    return {"clip": clip, "t": t, "path": to_sim(path), "grip": grip,
            "cube": to_sim(d["cube_start"])[0], "target": to_sim(d["target"])[0], "mapping": mapping}


# -----path post-processing
def contact_fixes(path, grip, cube_xy, target_xy):
    """Make the approach to each contact come from above, and keep the path off the table.
    Returns the fixed path and how many points were changed."""
    p = path.copy()
    top_safe = CUBE_HALF * 2 + CLEARANCE + SIM_TABLE_Z           
    changes = np.zeros(len(p), bool)
    g_k = int(np.argmax(grip)) if grip.any() else None              # first closed step = grasp
    r_k = g_k + int(np.argmin(grip[g_k:])) if g_k is not None and not grip[g_k:].all() else None
    for contact_k, xy in [(g_k, cube_xy), (r_k, target_xy)]:
        if contact_k is None:
            continue
        k0 = max(0, contact_k - VERTICAL_STEPS)
        lo = 0 if xy is cube_xy else g_k
        for k in range(lo, k0):                                     # don't skim low over the object
            if np.linalg.norm(p[k, :2] - xy) < CLEAR_RADIUS and p[k, 2] < top_safe:
                p[k, 2] = top_safe
                changes[k] = True
        above = np.r_[p[contact_k, :2], max(top_safe, p[k0, 2])]    # final descent: straight down
        for j, k in enumerate(range(k0, contact_k)):
            a = j / max(1, contact_k - k0)
            p[k] = above * (1 - a) + p[contact_k] * a
            changes[k] = True
        k1 = min(len(p), contact_k + VERTICAL_STEPS)                # after the contact: straight up
        for j, k in enumerate(range(contact_k + 1, k1)):
            a = (j + 1) / max(1, k1 - contact_k - 1)
            lifted = np.r_[p[contact_k, :2], max(top_safe, p[min(k1, len(p) - 1), 2])]
            new = p[contact_k] * (1 - a) + lifted * a
            if not np.allclose(new, p[k]):
                p[k] = new
                changes[k] = True
    low = p[:, 2] < MIN_Z + SIM_TABLE_Z
    p[low, 2] = MIN_Z + SIM_TABLE_Z
    return p, int((changes | low).sum())


# -----one episode
def move_action(eef, target, grip, gain=GAIN):
    a = np.zeros(7)
    a[:3] = np.clip(gain * (target - eef) / MAX_STEP, -1, 1)
    a[6] = 1.0 if grip else -1.0
    return a


def run_episode(path, grip, cube, target, seed=0, fixes=True, grip_pause=0, save_gif=False,
                est_cube=None, est_target=None):
    """Follow `path` (sim coordinates, 20 Hz) with the cube and target placed at `cube`, `target`.
    est_cube / est_target: where the robot BELIEVES the objects are (used by the contact fixes);
    default = the true positions. Returns (result dict, GIF frames)."""

    n_fixed = 0
    est_cube = cube if est_cube is None else est_cube
    est_target = target if est_target is None else est_target
    if fixes:
        path, n_fixed = contact_fixes(path, grip, np.asarray(est_cube)[:2], np.asarray(est_target)[:2])
    env = CubeToTarget(
        robots="Panda", has_renderer=False,
        has_offscreen_renderer=save_gif, use_camera_obs=save_gif,
        camera_names=CAMERA, camera_heights=256, camera_widths=256, control_freq=CONTROL_HZ,
        cube_x_range=(cube[0], cube[0]), cube_y_range=(cube[1], cube[1]),
        target_xy=(target[0], target[1]),
        seed=seed,                    # robosuite adds small random noise to the robot's start pose
    )
    obs = env.reset()
    frames = []

    def step(action):
        nonlocal obs
        obs, _, _, _ = env.step(action)
        if save_gif and len(frames) < 2000:
            frames.append(obs[f"{CAMERA}_image"].copy())      # copy: robosuite reuses the buffer

    for _ in range(10):                                        # let the cube settle
        step(np.r_[np.zeros(6), -1.0])
    rest_z = obs["cube_pos"][2]

    for _ in range(200):                                       
        if np.linalg.norm(path[0] - obs["robot0_eef_pos"]) < PREROLL_TOL:
            break
        step(move_action(obs["robot0_eef_pos"], path[0], False))

    max_z, track_err, prev = rest_z, [], grip[0]
    for k in range(len(path)):
        if grip[k] != prev and grip_pause:
            hold = obs["robot0_eef_pos"].copy()
            for _ in range(grip_pause):
                step(move_action(obs["robot0_eef_pos"], hold, grip[k]))
                max_z = max(max_z, obs["cube_pos"][2])
        prev = grip[k]
        step(move_action(obs["robot0_eef_pos"], path[k], grip[k]))
        track_err.append(np.linalg.norm(path[k] - obs["robot0_eef_pos"]))
        max_z = max(max_z, obs["cube_pos"][2])
    for _ in range(SETTLE_STEPS):
        step(move_action(obs["robot0_eef_pos"], obs["robot0_eef_pos"], False))

    lifted = bool(max_z > rest_z + LIFT_THRESHOLD)
    placed = bool(env._check_success())
    cube_final = obs["cube_pos"]
    env.close()
    return {
        "seed": seed, "success": lifted and placed, "lifted": lifted, "placed": placed,
        "max_lift_cm": round(float(max_z - rest_z) * 100, 1),
        "cube_to_target_cm": round(float(np.linalg.norm(cube_final[:2] - np.asarray(target)[:2])) * 100, 1),
        "tracking_error_cm_median": round(float(np.median(track_err)) * 100, 2),
        "tracking_error_cm_max": round(float(np.max(track_err)) * 100, 2),
        "contact_fixes": fixes, "points_changed": n_fixed, "points_total": len(path),
        "grip_pause_steps": grip_pause,
        "path_z_range_cm": [round(float((path[:, 2].min() - SIM_TABLE_Z) * 100), 1),
                            round(float((path[:, 2].max() - SIM_TABLE_Z) * 100), 1)],
    }, frames
