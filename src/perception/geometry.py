"""Camera geometry: marker position, pixel rays, ray-plane intersection, hand-length depth.

Conventions
  camera frame: x right, y down, z forward (OpenCV)
  table frame:  the ArUco marker's frame; x, y on the table, z UP out of the table, origin at the
                marker centre. A camera point p maps to the table as  R^T (p - t).
"""
import cv2
import numpy as np

WRIST, MIDDLE_BASE = 0, 9


def load_calibration(path):
    c = np.load(path)
    return c["K"], c["dist"]


def marker_pose(corners_all, K, dist, marker_length):
    """One position for the clip from the median of the detected corners.
    Returns R, t and the reprojection errors of the chosen and the alternative
    solution (square markers can have an ambiguous tilt)."""

    ok = ~np.isnan(corners_all[:, 0, 0])
    if ok.sum() == 0:
        raise ValueError("marker never detected in this clip")
    corners = np.median(corners_all[ok], axis=0).astype(np.float64)
    h = marker_length / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    best = int(np.argmin(np.ravel(errs)))
    R, _ = cv2.Rodrigues(rvecs[best])
    errs = sorted(float(e) for e in np.ravel(errs))
    return R, tvecs[best].ravel(), {"frames": int(ok.sum()), "reproj_px": errs[0],
                                    "alt_reproj_px": errs[1] if len(errs) > 1 else None}


def to_table(p_cam, R, t):
    """Camera-frame points to table frame."""

    return (np.atleast_2d(p_cam) - t) @ R


def camera_centre_table(R, t):
    return -R.T @ t


def undistort(px, K, dist):
    """Distorted pixels to ideal pinhole pixels."""

    px = np.asarray(px, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(px, K, dist, P=K).reshape(-1, 2)


def pixel_rays_cam(px, K, dist):
    """Unit ray direction in the camera frame for pixels."""

    u = undistort(px, K, dist)
    d = np.column_stack([(u[:, 0] - K[0, 2]) / K[0, 0], (u[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(u))])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def ray_plane(px, K, dist, R, t, z_plane):
    """Where the ray through each pixel hits the horizontal plane z = z_plane.
    z = 0 for things lying on the table, z = cube height for the cube's top face."""

    d_tab = pixel_rays_cam(px, K, dist) @ R          
    c = camera_centre_table(R, t)
    s = (z_plane - c[2]) / d_tab[:, 2]
    return c + s[:, None] * d_tab


def pinch_cam_hand_length(hand_px, pinch_px, K, dist, hand_length):
    """Method A for every frame: depth Z = f * hand_length / pixel length of wrist->middle-base,
    then the pinch pixel back-projected to that depth. NaN where the hand is missing."""

    n = len(pinch_px)
    out = np.full((n, 3), np.nan)
    ok = ~np.isnan(pinch_px[:, 0]) & ~np.isnan(hand_px[:, WRIST, 0])
    if not ok.any():
        return out
    f = (K[0, 0] + K[1, 1]) / 2
    w = undistort(hand_px[ok, WRIST], K, dist)
    m = undistort(hand_px[ok, MIDDLE_BASE], K, dist)
    p = undistort(pinch_px[ok], K, dist)
    Z = f * hand_length / np.linalg.norm(w - m, axis=1)
    out[ok] = np.column_stack([(p[:, 0] - K[0, 2]) / K[0, 0] * Z, (p[:, 1] - K[1, 2]) / K[1, 1] * Z, Z])
    return out


def draw_table_axes(frame, K, dist, R, t, length=0.05):
    """Draw the table's axes (x red, y green, z blue) at the marker."""

    rvec, _ = cv2.Rodrigues(R)
    cv2.drawFrameAxes(frame, K, dist, rvec, t, length, 3)
    return frame
