"""Expert data collection: run a generated path with the tracking controller and log every step.

Noise injection (DART): the robot executes the expert action plus small random noise, but the dataset
records the CLEAN action the expert intended. The robot drifts slightly off the path, so the data
contains examples of getting back on track from off-path states, the situations a learned policy
meets after its own small mistakes (or after a nudge). The tracking controller provides the correct
corrective action from any position, so the labels stay valid.

State-based contacts: a time-indexed expert closes the gripper at a fixed STEP, but the robot lags
its path slightly, so at that step it may be at the grasp point in one episode and 1-2 cm above it in
another. Near-identical states then carry contradictory labels (open / close), and a policy, which has
no clock, learns to hover. With state_contacts=True the expert closes only when the gripper has
actually REACHED the grasp point (and opens only at the release point), so the gripper label is a
function of the state the policy sees.
"""


import numpy as np
from src.sim.cube_env import CubeToTarget
from src.sim.generator import contact_indices
from src.policy.features import obs_features
from src.sim.replay import (CONTROL_HZ, LIFT_THRESHOLD, PREROLL_TOL, SETTLE_STEPS,
                            contact_fixes, move_action)

REACH_TOL = 0.008          # "reached the contact point" = within 8 mm
MAX_REACH_STEPS = 40       # give up waiting after 2 s (then act anyway; counted as a fallback)
GRIP_STEPS = 8             # hold at the contact point while the fingers close / open (0.4 s)


def collect_episode(path, grip, cube, target, seed=0, action_noise=0.1, fixes=True, state_contacts=True):
    """Run one generated path; return (log dict, success bool, info)."""
    if fixes:
        path, _ = contact_fixes(path, grip, np.asarray(cube)[:2], np.asarray(target)[:2])
    rng = np.random.default_rng(seed + 10_000)
    env = CubeToTarget(
        robots="Panda", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        control_freq=CONTROL_HZ,
        cube_x_range=(cube[0], cube[0]), cube_y_range=(cube[1], cube[1]),
        target_xy=(target[0], target[1]), seed=seed,
    )
    obs = env.reset()
    target_pos = env.target_pos.copy()
    log = {"obs": [], "act_move": [], "act_grip": [], "eef": [], "cube": [], "gripper_qpos": []}

    def step(expert_action):
        """Log the observation and the CLEAN expert action, then execute a noisy version."""
        nonlocal obs
        log["obs"].append(obs_features(obs, target_pos))
        log["act_move"].append(expert_action[:3].astype(np.float32).copy())
        log["act_grip"].append(np.float32(expert_action[6] > 0))
        log["eef"].append(obs["robot0_eef_pos"].copy())
        log["cube"].append(obs["cube_pos"].copy())
        log["gripper_qpos"].append(obs["robot0_gripper_qpos"].copy())
        executed = expert_action.copy()
        if action_noise:
            executed[:3] = np.clip(executed[:3] + rng.normal(0, action_noise, 3), -1, 1)
        obs, _, _, _ = env.step(executed)

    for _ in range(10):                                         # cube settles (not logged)
        obs, _, _, _ = env.step(np.r_[np.zeros(6), -1.0])
    rest_z = obs["cube_pos"][2]
    max_z = rest_z

    for _ in range(200):                                        # pre-roll to the start of the path
        if np.linalg.norm(path[0] - obs["robot0_eef_pos"]) < PREROLL_TOL:
            break
        step(move_action(obs["robot0_eef_pos"], path[0], False))
    g_k, r_k = contact_indices(grip)
    fallbacks = 0

    def reach_then_switch(point, holding_before, holding_after):
        """Move to the contact point (gripper unchanged) until reached, then switch the gripper there."""
        nonlocal fallbacks, max_z
        for _ in range(MAX_REACH_STEPS):
            if np.linalg.norm(point - obs["robot0_eef_pos"]) < REACH_TOL:
                break
            step(move_action(obs["robot0_eef_pos"], point, holding_before))
        else:
            fallbacks += 1
        for _ in range(GRIP_STEPS):
            step(move_action(obs["robot0_eef_pos"], point, holding_after))
            max_z = max(max_z, obs["cube_pos"][2])

    for k in range(len(path)):                                  # follow the generated path
        if state_contacts and k == g_k:
            reach_then_switch(path[g_k], False, True)           # grasp at the grasp POINT, not the grasp TIME
        if state_contacts and k == r_k:
            reach_then_switch(path[r_k], True, False)           # release at the release point
        step(move_action(obs["robot0_eef_pos"], path[k], grip[k]))
        max_z = max(max_z, obs["cube_pos"][2])
    for _ in range(SETTLE_STEPS):                               # hold still, gripper open
        step(move_action(obs["robot0_eef_pos"], obs["robot0_eef_pos"], False))

    success = bool(max_z > rest_z + LIFT_THRESHOLD) and bool(env._check_success())
    env.close()
    log = {k: np.array(v) for k, v in log.items()}
    return log, success, {"target": target_pos, "steps": len(log["obs"]), "fallbacks": fallbacks}
