"""Check a recorded clip before the main recording session.

Confirms, on real footage, that:
  1. the ArUco marker (DICT_4X4_50, ID 0) is detected,
  2. MediaPipe finds the hand while it's in view,
  3. the cube's blue top and the green target segment cleanly in HSV.

Usage:  python scripts/check_test_clip.py data/raw/test_clip.mov
Writes annotated frames and mask images to outputs/test_clip_check/.
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
OUT_DIR = PROJECT_ROOT / "outputs" / "test_clip_check"

# ---- ArUco
ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_ID = 0

# ---- HSV ranges (OpenCV scale: H 0-179, S 0-255, V 0-255). Tune these from the mask images.
BLUE_LOW, BLUE_HIGH = (85, 40, 100), (115, 255, 255)       # cube top face
GREEN_LOW, GREEN_HIGH = (50, 100, 60), (80, 255, 255)     # target square (#14B428)
MIN_BLOB_AREA = 300        # pixels; smaller blobs are treated as noise

N_SNAPSHOTS = 8            # annotated frames to save, spread across the clip
THUMB_TIP, INDEX_TIP = 4, 8


def largest_blob(mask):
    """Area and contour of the largest connected region in a binary mask."""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, None
    biggest = max(contours, key=cv2.contourArea)
    return cv2.contourArea(biggest), biggest


def colour_mask(hsv, low, high):
    mask = cv2.inRange(hsv, np.array(low), np.array(high))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove speckles
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes


def median_hsv(hsv, contour):
    """Median H, S, V inside a contour: use these to tune the ranges above."""
    
    region = np.zeros(hsv.shape[:2], np.uint8)
    cv2.drawContours(region, [contour], -1, 255, thickness=-1)
    return np.median(hsv[region > 0], axis=0).astype(int)


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (0, 0, 255), -1)
    pinch = ((pts[THUMB_TIP][0] + pts[INDEX_TIP][0]) // 2, (pts[THUMB_TIP][1] + pts[INDEX_TIP][1]) // 2)
    cv2.circle(frame, pinch, 7, (0, 255, 255), 2)               # pinch midpoint = gripper point


def main(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video_path.name}: {width}x{height}, {fps:.1f} fps, {n_frames} frames ({n_frames / fps:.1f} s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_frames = set(np.linspace(0, max(n_frames - 1, 0), N_SNAPSHOTS).astype(int))

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                                       cv2.aruco.DetectorParameters())
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )

    rows = []
    with vision.HandLandmarker.create_from_options(options) as hands:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # 1. ArUco
            corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            marker_found = ids is not None and MARKER_ID in ids.flatten()

            # 2. MediaPipe (VIDEO mode needs a strictly increasing timestamp in ms)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                            int(i * 1000 / fps))
            hand_found = len(result.hand_landmarks) > 0

            # 3. Colour masks
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue_mask = colour_mask(hsv, BLUE_LOW, BLUE_HIGH)
            green_mask = colour_mask(hsv, GREEN_LOW, GREEN_HIGH)
            blue_area, blue_contour = largest_blob(blue_mask)
            green_area, green_contour = largest_blob(green_mask)

            rows.append((marker_found, hand_found, blue_area, green_area))

            if i == 0:   # mask images from the first (empty-scene) frame, for HSV tuning
                cv2.imwrite(str(OUT_DIR / "masks_first_frame.png"),
                            np.hstack([frame, cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR),
                                       cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)]))
                for name, area, contour in [("blue (cube)", blue_area, blue_contour),
                                            ("green (target)", green_area, green_contour)]:
                    if contour is not None and area >= MIN_BLOB_AREA:
                        print(f"  first frame {name}: area {area:.0f} px, median HSV {median_hsv(hsv, contour)}")
                    else:
                        print(f"  first frame {name}: NOT FOUND - widen its HSV range")

            if i in snapshot_frames:
                vis = frame.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                if hand_found:
                    draw_hand(vis, result.hand_landmarks[0])
                for contour, area, colour in [(blue_contour, blue_area, (255, 0, 0)),
                                              (green_contour, green_area, (0, 255, 0))]:
                    if contour is not None and area >= MIN_BLOB_AREA:
                        cv2.drawContours(vis, [contour], -1, colour, 3)
                label = f"frame {i}  marker:{'Y' if marker_found else 'N'}  hand:{'Y' if hand_found else 'N'}"
                cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imwrite(str(OUT_DIR / f"frame_{i:04d}.png"), vis)
            i += 1
    cap.release()

    rows = np.array(rows, dtype=float)
    if len(rows) == 0:
        sys.exit("No frames could be read from the video.")
    marker_rate = rows[:, 0].mean()
    hand_rate = rows[:, 1].mean()
    cube_rate = (rows[:, 2] >= MIN_BLOB_AREA).mean()
    target_rate = (rows[:, 3] >= MIN_BLOB_AREA).mean()

    print(f"\nFrames processed: {len(rows)}")
    print(f"  Marker ID {MARKER_ID} detected:   {marker_rate:6.1%}   (want ~100%; it should never be covered)")
    print(f"  Hand detected:            {hand_rate:6.1%}   (should match the time your hand is in view)")
    print(f"  Cube (blue) visible:      {cube_rate:6.1%}   (dips while the hand covers it are expected)")
    print(f"  Target (green) visible:   {target_rate:6.1%}   (want ~100%)")
    print(f"\nAnnotated frames and masks saved to {OUT_DIR}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "raw" / "test_clip.mov"
    main(path)