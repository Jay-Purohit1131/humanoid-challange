"""Camera calibration from a video of the printed checkerboard.

Finds the board in frames sampled across the video, keeps sharp and well-spread views,
runs OpenCV's calibration, removes outlier views, and saves the camera matrix and
distortion coefficients to data/calib/camera.npz.

Usage:  python scripts/calibrate_camera.py data/raw/calib.mov
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "calib"

# ---- Board: 10 x 7 squares -> 9 x 6 INNER corners (OpenCV counts inner corners)
PATTERN_SIZE = (9, 6)
SQUARE_SIZE_M = 0.020          

# ---- Frame selection
SAMPLE_EVERY_S = 0.25          
MAX_VIEWS = 40                 
MIN_VIEWS = 12                 
BLUR_THRESHOLD = 30.0          
OUTLIER_FACTOR = 2.0           
OUTLIER_FLOOR_PX = 0.3         

CALIB_FLAGS = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
               | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
INITIAL_FOCAL_PX = 1700.0       

# ---- Stillness filter: only use frames where the board wasn't moving
STILL_GAP_FRAMES = 3          
STILL_MAX_PX = 1.0             


def board_object_points():
    """3D corner positions in the board's own frame (z = 0), in metres."""
    grid = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    grid[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    return grid * SQUARE_SIZE_M


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def detect_board(gray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)   # refine to sub-pixel


def initial_K(image_size):
    """Starting camera matrix: guessed focal length, principal point at the image centre."""
    w, h = image_size
    return np.array([[INITIAL_FOCAL_PX, 0, w / 2], [0, INITIAL_FOCAL_PX, h / 2], [0, 0, 1]], dtype=np.float64)


def per_view_errors(obj_pts, img_pts, K, dist, rvecs, tvecs):
    """RMS reprojection error (pixels) for each view."""
    errors = []
    for obj, img, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, r, t, K, dist)
        errors.append(np.sqrt(np.mean(np.sum((img - projected) ** 2, axis=2))))
    return np.array(errors)


def coverage(img_pts, image_size, cells=3):
    """Which cells of a 3x3 grid over the image contain detected corners."""
    w, h = image_size
    grid = np.zeros((cells, cells), bool)
    for pts in img_pts:
        p = pts.reshape(-1, 2)
        cx = np.clip((p[:, 0] / w * cells).astype(int), 0, cells - 1)
        cy = np.clip((p[:, 1] / h * cells).astype(int), 0, cells - 1)
        grid[cy, cx] = True
    return grid


def main(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * SAMPLE_EVERY_S)))

    detections, image_size, i = [], None, 0
    n_tried = n_blurry = n_found = n_moving = 0
    pending = None          # (frame index, sharpness, corners) waiting for its stillness check
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Stillness check: STILL_GAP_FRAMES after a detection, find the board again and compare
        if pending is not None and i == pending[0] + STILL_GAP_FRAMES:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            later = detect_board(gray)
            if later is not None:
                motion = np.mean(np.linalg.norm(later - pending[2], axis=2))
                if motion < STILL_MAX_PX:
                    detections.append(pending)
                else:
                    n_moving += 1
            else:
                n_moving += 1           # board lost a few frames later: almost certainly moving
            pending = None

        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]               # (width, height)
            n_tried += 1
            sharp = sharpness(gray)
            if sharp < BLUR_THRESHOLD:
                n_blurry += 1
            else:
                corners = detect_board(gray)
                if corners is not None:
                    n_found += 1
                    pending = (i, sharp, corners)
        i += 1
    cap.release()

    print(f"{video_path.name}: {image_size[0]}x{image_size[1]}, {i} frames")
    print(f"  sampled {n_tried} frames: {n_found} with the board found, {n_blurry} too blurry")
    print(f"  stillness filter: {len(detections)} still, {n_moving} rejected as moving")
    if len(detections) < MIN_VIEWS:
        sys.exit(f"Only {len(detections)} usable views (need {MIN_VIEWS}+). Re-record: slower, more pauses, whole board in frame.")

    # Spread the chosen views across the whole video
    if len(detections) > MAX_VIEWS:
        keep = np.linspace(0, len(detections) - 1, MAX_VIEWS).astype(int)
        detections = [detections[k] for k in keep]

    obj = board_object_points()
    img_pts = [d[2] for d in detections]
    obj_pts = [obj] * len(img_pts)

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, initial_K(image_size), None, flags=CALIB_FLAGS)
    errors = per_view_errors(obj_pts, img_pts, K, dist, rvecs, tvecs)
    print(f"\nFirst pass: {len(img_pts)} views, RMS reprojection error {rms:.3f} px")

    # Drop outlier views and recalibrate once
    good = errors <= max(OUTLIER_FACTOR * np.median(errors), OUTLIER_FLOOR_PX)
    if not good.all() and good.sum() >= MIN_VIEWS:
        img_pts = [p for p, g in zip(img_pts, good) if g]
        obj_pts = [obj] * len(img_pts)
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, initial_K(image_size), None, flags=CALIB_FLAGS)
        print(f"Removed {(~good).sum()} outlier view(s) -> {len(img_pts)} views, RMS {rms:.3f} px")

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = image_size
    hfov = np.degrees(2 * np.arctan(w / (2 * fx)))
    cov = coverage(img_pts, image_size)

    print("\nResults")
    print(f"  RMS reprojection error: {rms:.3f} px   ({'good' if rms < 0.5 else 'acceptable' if rms < 1.0 else 'poor: re-record'})")
    print(f"  fx, fy = {fx:.1f}, {fy:.1f} px   (should be nearly equal; ratio {fx / fy:.4f})")
    print(f"  cx, cy = {cx:.1f}, {cy:.1f} px   (image centre is {w / 2:.0f}, {h / 2:.0f})")
    print(f"  horizontal field of view: {hfov:.1f} deg")
    print(f"  distortion (k1, k2, p1, p2, k3): {np.round(dist.ravel(), 4)}")
    print("  image coverage (3x3 grid, # = corners seen there):")
    for row in cov:
        print("    " + " ".join("#" if c else "." for c in row))
    if not cov.all():
        print("  -> some image regions have no corners; distortion near those edges is less reliable")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "camera.npz", K=K, dist=dist, image_size=np.array(image_size),
             rms=rms, n_views=len(img_pts), square_size_m=SQUARE_SIZE_M)
    with open(OUT_DIR / "camera.json", "w") as f:
        json.dump({"K": K.tolist(), "dist": dist.ravel().tolist(), "image_size": list(image_size),
                   "rms_px": float(rms), "n_views": len(img_pts), "square_size_m": SQUARE_SIZE_M,
                   "source_video": video_path.name}, f, indent=2)

    # Visual check: one frame before and after undistortion
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, detections[len(detections) // 2][0])
    ok, frame = cap.read()
    cap.release()
    if ok:
        undistorted = cv2.undistort(frame, K, dist)
        cv2.imwrite(str(OUT_DIR / "undistort_check.jpg"), np.hstack([frame, undistorted]))

    print(f"\nSaved {OUT_DIR / 'camera.npz'}, camera.json and undistort_check.jpg")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "raw" / "calib.mov"
    main(path)