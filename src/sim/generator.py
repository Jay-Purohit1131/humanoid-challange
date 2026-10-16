"""Object-relative trajectory generation (the MimicGen idea, position-only).

A recorded demo is split at the grasp and the release:
  approach  (start .. grasp)    belongs to the CUBE    -> shifted by how far the cube moved
  carry     (grasp .. release)  hands over from the cube to the TARGET
  retreat   (release .. end)    belongs to the TARGET  -> shifted by how far the target moved

Because the grasp is top-down and the cube's orientation is fixed, MimicGen's pose transforms
(T_new @ inv(T_old)) reduce to translations. The carry blends the two shifts smoothly, so the
generated path has no jumps: it starts at the new cube and ends at the new target.
"""
import numpy as np


def contact_indices(grip):
    """First closed step (grasp) and the first open step after it (release)."""
    g = int(np.argmax(grip))
    r = g + int(np.argmin(grip[g:]))
    return g, r


def smoothstep(u):
    """0 -> 1 with zero slope at both ends so no sudden change of velocity at the handover."""

    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def generate(path, grip, old_cube, old_target, new_cube, new_target):
    """Re-attach the recorded path to new cube and target positions. Only x, y shift; 
    heights are kept the same."""

    g, r = contact_indices(grip)
    d_cube = np.r_[np.asarray(new_cube)[:2] - np.asarray(old_cube)[:2], 0.0]
    d_target = np.r_[np.asarray(new_target)[:2] - np.asarray(old_target)[:2], 0.0]

    new = path.copy()
    new[:g] += d_cube                                    # approach follows the cube
    a = smoothstep((np.arange(g, r) - g) / max(1, r - g))
    new[g:r] += (1 - a)[:, None] * d_cube + a[:, None] * d_target   # cube to target handover
    new[r:] += d_target                                  # place and retreat follows the target
    return new


def sample_scenes(cube, target, n, seed=0, cube_range=(0.08, 0.06), target_range=(0.08, 0.06),
                  min_separation=0.15):
    """n random (cube, target) positions around the recorded ones, at least min_separation apart.
    Fixed seed, so every method is tested on exactly the same scenes."""

    rng = np.random.default_rng(seed)
    scenes = []
    while len(scenes) < n:
        c = np.asarray(cube, float).copy()
        t = np.asarray(target, float).copy()
        c[:2] += rng.uniform(-1, 1, 2) * np.asarray(cube_range)
        t[:2] += rng.uniform(-1, 1, 2) * np.asarray(target_range)
        if np.linalg.norm(c[:2] - t[:2]) >= min_separation:
            scenes.append((c, t))
    return scenes
