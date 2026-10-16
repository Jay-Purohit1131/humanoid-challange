"""Stage 1 of perception: extract raw pixel detections from every frame of a clip.

For each frame it saves the marker corners, the hand landmarks (pixels + MediaPipe world
landmarks), the pinch midpoint, and the centres of the blue cube top and the green target.
Missing detections are stored as NaN, so every array has exactly one row per frame.

Needs no camera calibration, so it only has to run once per clip; the metric
(calibration-dependent) geometry is done later from the saved file.

Usage:  python scripts/extract_2d.py data/raw/clean_01.mp4 
Output: data/processed/<clip>_2d.npz
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

MARKER_ID = 0
BLUE_LOW, BLUE_HIGH = (85, 40, 100), (115, 255, 255)      # light-blue cube top 
GREEN_LOW, GREEN_HIGH = (50, 60, 60), (80, 255, 255)      # target square
MIN_BLOB_AREA = 3000                                      # pixels
THUMB_TIP, INDEX_TIP = 4, 8


def colour_mask(hsv, low, high):
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove speckles
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes


def largest_blob(mask):
    """Area and contour of the largest connected region in a binary mask."""

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, None
    biggest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(biggest), biggest


def blob_info(contour):
    """Centre (x, y), rotation angle (deg), 4 corner points and (w, h) of the blob's rotated rectangle."""
    
    M = cv2.moments(contour)
    centre = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])   # mean x, mean y of the blob
    rect = cv2.minAreaRect(contour)                                  # ((cx, cy), (w, h), angle)
    return centre, cv2.boxPoints(rect)


def find_colour(hsv, low, high):
    """(centre, area, angle, corners, rect w/h) of the largest blob in a colour range, or None."""
    
    area, contour = largest_blob(colour_mask(hsv, low, high))
    if contour is None or area < MIN_BLOB_AREA:
        return None
    centre, corners = blob_info(contour)
    return centre, area, corners


def extract(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}, skipping")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                       cv2.aruco.DetectorParameters())
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    keys = ["t", "marker_corners", "hand_px", "hand_world", "pinch_px",
            "cube_px", "cube_area", "target_px", "target_box"]
    out = {k: [] for k in keys}                 # one list per field, one entry per frame
    nan2 = np.full(2, np.nan)

    with vision.HandLandmarker.create_from_options(options) as hands:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t_ms = int(round(i * 1000 / fps))   # VIDEO mode needs strictly increasing ms timestamps
            out["t"].append(t_ms / 1000)

            # ---- 1. Marker (ID 0): its 4 corners in pixels
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if ids is not None and MARKER_ID in ids.flatten():
                k = int(np.where(ids.flatten() == MARKER_ID)[0][0])
                out["marker_corners"].append(corners[k][0])          # (1, 4, 2) -> (4, 2)
            else:
                out["marker_corners"].append(np.full((4, 2), np.nan))

            # ---- 2. Hand: landmarks are normalised (0-1), so scale to pixels
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
            if result.hand_landmarks:
                px = np.array([[lm.x * width, lm.y * height] for lm in result.hand_landmarks[0]])
                world = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_world_landmarks[0]])  # metres
                out["hand_px"].append(px)
                out["hand_world"].append(world)
                out["pinch_px"].append((px[THUMB_TIP] + px[INDEX_TIP]) / 2)   # the gripper point
            else:
                out["hand_px"].append(np.full((21, 2), np.nan))
                out["hand_world"].append(np.full((21, 3), np.nan))
                out["pinch_px"].append(nan2)

            # ---- 3. Cube (blue top face) and target (green square)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cube = find_colour(hsv, BLUE_LOW, BLUE_HIGH)
            if cube is not None:
                centre, area, _ = cube
                out["cube_px"].append(centre)
                out["cube_area"].append(area)
                
            else:
                out["cube_px"].append(nan2)
                out["cube_area"].append(0.0)
                

            target = find_colour(hsv, GREEN_LOW, GREEN_HIGH)
            if target is not None:
                centre, _, box = target
                out["target_px"].append(centre)
                out["target_box"].append(box)
            else:
                out["target_px"].append(nan2)
                out["target_box"].append(np.full((4, 2), np.nan))

            i += 1
    cap.release()

    data = {k: np.array(v, dtype=float) for k, v in out.items()}   # first axis = frame
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{video_path.stem}_2d.npz"
    np.savez_compressed(out_path, fps=fps, image_size=np.array([width, height]),
                        source=str(video_path.name), **data)

    def found(arr):            # fraction of frames where the detection exists
        return np.mean(~np.isnan(arr.reshape(len(arr), -1)[:, 0]))

    print(f"{video_path.name}: {len(data['t'])} frames | marker {found(data['marker_corners']):.0%} | "
          f"hand {found(data['pinch_px']):.0%} | cube {found(data['cube_px']):.0%} | "
          f"target {found(data['target_px']):.0%} -> {out_path.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python scripts/extract_2d.py <clip> [more clips...]")
    for path in sys.argv[1:]:
        extract(Path(path))
