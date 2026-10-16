"""What the policy sees, computed the same way during data collection and during evaluation.

Relative vectors ("where is the cube from my gripper") generalise across scenes much better than
absolute coordinates, because the task becomes "move towards the thing" wherever the thing is.
"""
import numpy as np

SIM_TABLE_Z = 0.8
FEATURE_NAMES = [
    "cube_minus_eef_x", "cube_minus_eef_y", "cube_minus_eef_z",
    "target_minus_eef_x", "target_minus_eef_y", "target_minus_eef_z",
    "target_minus_cube_x", "target_minus_cube_y", "target_minus_cube_z",
    "gripper_opening",          # distance between the fingers (m): ~0.08 open, ~0.056 on the cube
    "eef_height",               # gripper height above the table (m)
]


def obs_features(obs, target_pos):
    """11 numbers from a robosuite observation dict and the target's position."""
    eef = obs["robot0_eef_pos"]
    cube = obs["cube_pos"]
    target = np.asarray(target_pos, dtype=float)
    q = obs["robot0_gripper_qpos"]
    return np.concatenate([
        cube - eef,
        target - eef,
        target - cube,
        [q[0] - q[1]],
        [eef[2] - SIM_TABLE_Z],
    ]).astype(np.float32)
