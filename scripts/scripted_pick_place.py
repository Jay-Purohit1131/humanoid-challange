"""Scripted pick-and-place in the CubeToTarget scene, using hard-coded waypoints.
 
A proportional controller moves the gripper towards each waypoint in turn; gripper
phases hold position while the fingers close or open. Optionally saves a GIF.
"""
import sys
from pathlib import Path
 
import imageio.v3 as iio
import numpy as np
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
 
import robosuite.macros as macros  # noqa: E402
macros.IMAGE_CONVENTION = "opencv"   # images the right way up 
 
from src.sim.cube_env import CubeToTarget  # noqa: E402
 
# ---- Controller settings
MAX_STEP = 0.05             # metres of movement per unit of action (OSC_POSE output limit)
GAIN = 0.6                  # below 1 = slow down near the waypoint instead of overshooting
TOLERANCE = 0.01            # metres: "close enough" to a waypoint
MAX_STEPS_PER_PHASE = 200   # safety limit 
OPEN, CLOSE = -1.0, 1.0     # Panda gripper: -1 opens, +1 closes
 
# ---- Task geometry
HOVER = 0.10                # metres above the cube for the approach
LIFT = 0.15                 # metres to lift the cube
PLACE_CLEARANCE = 0.01      # release the cube this far above its resting height
GRIP_HOLD_STEPS = 15        # steps for the fingers to close/open (0.75 s at 20 Hz)
LIFT_THRESHOLD = 0.05       # cube must rise this far above rest to count as "lifted"
 
# ---- GIF output
SAVE_GIF = True
CAMERA = "frontview"
GIF_EVERY = 2               # keep every 2nd frame to keep the file small
GIF_PATH = PROJECT_ROOT / "outputs" / "gifs" / "scripted_pick_place.gif"
 
 
def move_action(eef_pos, waypoint, grip):
    """7-number OSC action that moves the gripper towards `waypoint`."""
    action = np.zeros(7)
    error = waypoint - eef_pos                                   # metres
    action[:3] = np.clip(GAIN * error / MAX_STEP, -1.0, 1.0)     # to action units
    action[6] = grip
    return action
 
 
env = CubeToTarget(
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=SAVE_GIF,
    use_camera_obs=SAVE_GIF,
    camera_names=CAMERA,
    camera_heights=256,
    camera_widths=256,
    control_freq=20,
)
obs = env.reset()
for _ in range(10):                                  # letting the cube settle
    obs, _, _, _ = env.step(np.zeros(env.action_dim))
 
cube = obs["cube_pos"].copy()                        # where the cube starts
target = env.target_pos.copy()                       # where it should end up
rest_z = env.cube_resting_z
 
# (name, waypoint, gripper, hold_steps). Waypoint None = gripper phase: hold still.
# robot0_eef_pos is the point between the fingertips, so grasp at the cube's centre height.

phases = [
    ("above_cube",   np.r_[cube[:2], cube[2] + HOVER],          OPEN,  None),
    ("descend",      np.r_[cube[:2], cube[2]],                  OPEN,  None),
    ("close",        None,                                      CLOSE, GRIP_HOLD_STEPS),
    ("lift",         np.r_[cube[:2], cube[2] + LIFT],           CLOSE, None),
    ("above_target", np.r_[target[:2], cube[2] + LIFT],         CLOSE, None),
    ("lower",        np.r_[target[:2], rest_z + PLACE_CLEARANCE], CLOSE, None),
    ("open",         None,                                      OPEN,  GRIP_HOLD_STEPS),
    ("retreat",      np.r_[target[:2], rest_z + HOVER],         OPEN,  None),
]
 
frames = []
max_cube_z = cube[2]
step_count = 0
 
 
def do_step(action):
    """Step the sim, track the cube's highest point, and collect GIF frames."""
    global obs, max_cube_z, step_count
    obs, _, _, _ = env.step(action)
    step_count += 1
    max_cube_z = max(max_cube_z, obs["cube_pos"][2])
    if SAVE_GIF and step_count % GIF_EVERY == 0:
        frames.append(obs[f"{CAMERA}_image"].copy())   # copy: robosuite may reuse the image buffer
 
 
for name, waypoint, grip, hold_steps in phases:
    if waypoint is None:
        hold_pos = obs["robot0_eef_pos"].copy()      # remember the position once, then hold it
        for _ in range(hold_steps):
            do_step(move_action(obs["robot0_eef_pos"], hold_pos, grip))
    else:
        for _ in range(MAX_STEPS_PER_PHASE):
            if np.linalg.norm(waypoint - obs["robot0_eef_pos"]) < TOLERANCE:
                break
            do_step(move_action(obs["robot0_eef_pos"], waypoint, grip))
 
    print(f"{name:13s} gripper {np.round(obs['robot0_eef_pos'], 3)}  "
          f"cube {np.round(obs['cube_pos'], 3)}  fingers {np.round(obs['robot0_gripper_qpos'], 3)}")
 
lifted = max_cube_z > rest_z + LIFT_THRESHOLD
placed = env._check_success()
print(f"\nSteps: {step_count} ({step_count / 20:.1f} s)   "
      f"Max cube height above rest: {(max_cube_z - rest_z) * 100:.1f} cm")
print(f"Lifted: {lifted}   Placed in target: {placed}   SUCCESS: {lifted and placed}")
 
if SAVE_GIF:
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(GIF_PATH, frames, duration=1000 * GIF_EVERY / 20, loop=0)   # duration per frame, ms
    print(f"Saved {len(frames)} frames to {GIF_PATH}")
 
env.close()